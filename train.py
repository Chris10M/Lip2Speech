#!/usr/bin/env python3

import torch
from apex import amp
from torch import optim as Optimizer
import numpy as np
torch.manual_seed(1)
from logger import setup_logger
import os
import collections
from torch.utils.data import DataLoader
import hashlib
import os
import math
import os.path as osp
import logging
import time
import datetime

from train_utils.tensorboard_logger import Tacotron2Logger
from datasets import train_collate_fn_pad, FaceAugmentation 
from datasets.grid import GRID
from datasets.wild import WILD
from datasets.lrw import LRW
from datasets.avspeech import AVSpeech
from evaluate import evaluate_net
from train_utils.losses import *
from model import model
from hparams import create_hparams
import arg_parser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Logger:
	logger = None
	ModelSavePath = 'savedmodels'
	tensor_board = None


def set_model_logger(net):
	model_info = str(net)


	respth = f'{Logger.ModelSavePath}/{hashlib.md5(model_info.encode()).hexdigest()}'
	Logger.ModelSavePath = respth

	if not osp.exists(respth): os.makedirs(respth)
	logger = logging.getLogger()

	if setup_logger(respth):
		logger.info(model_info)

	Logger.logger = logger

	tf_logs = f'{respth}/tf-logs'; os.makedirs(tf_logs, exist_ok=True)
	Logger.tensor_board = Tacotron2Logger(tf_logs)


def train(args):
	dataset_name = args.dataset
	dataset_path = args.dataset_path
	
	if dataset_name == 'LRW':
		ds = LRW(dataset_path, face_augmentation=FaceAugmentation())
		val_ds = LRW(dataset_path, mode='test', face_augmentation=FaceAugmentation())
	elif dataset_name == 'GRID':
		ds = GRID(dataset_path, face_augmentation=FaceAugmentation())
		val_ds = LRW(dataset_path, mode='test', face_augmentation=FaceAugmentation())
	elif dataset_name == 'AVSpeech':
		ds = AVSpeech(dataset_path, face_augmentation=FaceAugmentation())
		val_ds = LRW(dataset_path, mode='test', face_augmentation=FaceAugmentation())
	elif dataset_name == 'WILD':
		ds = WILD(dataset_path, face_augmentation=FaceAugmentation())
		val_ds = LRW(dataset_path, mode='test', face_augmentation=FaceAugmentation())
	else:
		assert "Dataset Not Present"
	
	saved_path = args.finetune_model

	
	hparams = create_hparams()
	
	net = model.get_network('train').to(device)
	set_model_logger(net)
	
	tf_ratio = 0.1
	max_iter = 6400000
	save_iter = 2000
	n_img_per_gpu = hparams.batch_size
	n_workers = min(n_img_per_gpu, os.cpu_count())
	
	dl = DataLoader(ds,
					batch_size=n_img_per_gpu,
					shuffle=False,
					num_workers=n_workers,
					pin_memory=True,
					drop_last=False, 
					collate_fn=train_collate_fn_pad)

	optim = Optimizer.AdamW([{'params': net.decoder.parameters()},
							 {'params': net.encoder.parameters()},
							], lr=hparams.learning_rate, weight_decay=hparams.weight_decay, amsgrad=True)

	if hparams.fp16_run:
		net, optim = amp.initialize(net, optim, opt_level='O2')

	max_eval_score = 0
	start_it = 0
	if os.path.isfile(saved_path):
		state_dict = torch.load(saved_path, map_location=device)
		if 'state_dict' in state_dict: state_dict = state_dict['state_dict']

		try:
			net.load_state_dict(state_dict, strict=False)
		except RuntimeError as e:
			print(e)
		
		try:
			start_it = 0
			start_it = state_dict['start_it'] + 2
		except KeyError:
			start_it = 0

		try:
			max_eval_score = state_dict['max_eval_score']
		except KeyError: ...

		try:
			optim.load_state_dict(state_dict['optimize_state'])
			...
		except Exception as e: print(e)


		print(f'Model Loaded: {saved_path} @ start_it: {start_it}')


	reconstruction_criterion = Loss()
	
	## train loop
	msg_iter = 50
	loss_log = collections.defaultdict(float)
	st = glob_st = time.time()
	diter = iter(dl)
	epoch = 0

	batch = next(diter)

	net = net.train()
	for it in range(start_it, max_iter):
		try:
			batch = next(diter)
		except StopIteration:
			epoch += 1
			diter = iter(dl)
			batch = next(diter)

			if epoch % 10 == 0:
				tf_ratio += 0.1

		(videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates), face_crops = batch
	
		videos, audios, melspecs, face_crops = videos.to(device), audios.to(device), melspecs.to(device), face_crops.to(device)
		video_lengths, audio_lengths, melspec_lengths = video_lengths.to(device), audio_lengths.to(device), melspec_lengths.to(device)
		mel_gates = mel_gates.to(device)
		outputs = net(videos, face_crops, audios, melspecs, video_lengths, audio_lengths, melspec_lengths, tf_ratio)
		
		
		losses = dict()

		losses = reconstruction_criterion(outputs, (melspecs, mel_gates), losses)

		loss = sum(losses.values())
		losses['loss'] = loss


		optim.zero_grad()

		if hparams.fp16_run:
			with amp.scale_loss(loss, optim) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()

		if hparams.fp16_run:
			grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optim), hparams.grad_clip_thresh)
			is_overflow = math.isnan(grad_norm)
		else:
			is_overflow = False
			grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), hparams.grad_clip_thresh)
		
		optim.step()
	
	
		if is_overflow: continue

		for k, v in losses.items(): loss_log[k] += v.item()
		if (it + 1) % save_iter == 0:
			save_pth = osp.join(Logger.ModelSavePath, f'{it + 1}_{int(time.time())}.pth')

			eval_score = evaluate_net(net, val_ds)
			
			Logger.logger.info(f"Model@{it + 1}\n Evaluation score: {eval_score}")
			Logger.tensor_board.log_validation(eval_score, net, (melspecs, mel_gates), outputs, it + 1)

			if eval_score < max_eval_score:  
				print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
				torch.save({
					'start_it': it,
					'state_dict': net.state_dict(),
					'optimize_state': optim.state_dict(),
					'max_eval_score': max_eval_score,
				}, save_pth)
				print(f'model at: {(it + 1)} Saved')

				max_eval_score = eval_score

		#   print training log message
		if (it+1) % msg_iter == 0:
			for k, v in loss_log.items(): loss_log[k] = round(v / msg_iter, 2)

			ed = time.time()
			t_intv, glob_t_intv = ed - st, ed - glob_st
			eta = int((max_iter - it) * (glob_t_intv / it))
			eta = str(datetime.timedelta(seconds=eta))
			msg = ', '.join([
					f'epoch: {epoch}',
					'it: {it}/{max_it}',         
					*[f"{k}: {v}" for k, v in loss_log.items()],
					f'tf_ratio: {tf_ratio}',
					'eta: {eta}',
					'time: {time:.2f}',
				]).format(
					it = it+1,
					max_it = max_iter,
					time = t_intv,
					eta = eta
				)
			Logger.tensor_board.log_training(loss_log['loss'], grad_norm, hparams.learning_rate, t_intv, it + 1)
			Logger.logger.info(msg)

			Logger.tensor_board.log_predictions(outputs, (melspecs, mel_gates))
			Logger.tensor_board.log_alignment(F.softmax(outputs[4], dim=-1), it + 1)
			
			loss_log = collections.defaultdict(float)
			st = ed

	save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
	net.cpu()
	torch.save({'state_dict': net.state_dict()}, save_pth)

	Logger.logger.info('training done, model saved to: {}'.format(save_pth))


def main():
	args = arg_parser.train()
	train(args)


if __name__ == "__main__":
	main()
