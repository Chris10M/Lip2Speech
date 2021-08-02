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
from train_utils.losses import *
from model import model
from hparams import create_hparams


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


def main():
	hparams = create_hparams()
	
	# ds = AVSpeech('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='test', face_augmentation=FaceAugmentation())
	# ds = GRID('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/GRID', mode='test', face_augmentation=FaceAugmentation())
	# ds = WILD('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/DL/Deep_Learning(CS7015)___Lec_3_2_A_typical_Supervised_Machine_Learning_Setup_uDcU3ZzH7hs_mp4', 
	# 		  mode='test', face_augmentation=FaceAugmentation(), duration=1.5)
	ds = LRW('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/LRW', face_augmentation=FaceAugmentation())

	net = model.get_network('train').to(device)
	set_model_logger(net)
	
	saved_path = ''
	
	tf_ratio = 0.95
	max_iter = 6400000
	save_iter = 2000
	n_img_per_gpu = 16
	n_workers = min(n_img_per_gpu, os.cpu_count())
	
	dl = DataLoader(ds,
					batch_size=n_img_per_gpu,
					shuffle=True,
					num_workers=n_workers,
					pin_memory=True,
					drop_last=False, 
					collate_fn=train_collate_fn_pad)

	optim = Optimizer.AdamW([{'params': net.decoder.parameters()},
							 {'params': net.encoder.parameters()},
							], lr=hparams.learning_rate, weight_decay=hparams.weight_decay, amsgrad=True)

	if hparams.fp16_run:
		net.decoder.decoder.attention_layer.score_mask_value = np.finfo('float16').min
		net, optim = amp.initialize(net, optim, opt_level='O2')

	min_eval_loss = 1e5
	start_it = 0
	if os.path.isfile(saved_path):
		loaded_model = torch.load(saved_path, map_location=device)
		state_dict = loaded_model['state_dict']

		try:
			net.load_state_dict(state_dict, strict=False)
		except RuntimeError as e:
			print(e)
		
		try:
			start_it = 0
			start_it = loaded_model['start_it'] + 2
		except KeyError:
			start_it = 0

		try:
			min_eval_loss = loaded_model['min_eval_loss']
		except KeyError: ...

		try:
			optim.load_state_dict(loaded_model['optimize_state'])
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

			# evaluation = evaluate_net(args, net)
			# Logger.logger.info(f"Model@{it + 1}\n{evaluation}"
			# optim.reduce_lr_on_plateau(eval_loss)
				eval_loss = loss
				Logger.tensor_board.log_validation(eval_loss, net, (melspecs, mel_gates), outputs, it + 1)

			# if eval_loss < min_eval_loss:  
				print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
				torch.save({
					'start_it': it,
					'state_dict': net.state_dict(),
					'optimize_state': optim.state_dict(),
					'min_eval_loss': min_eval_loss,
				}, save_pth)
				print(f'model at: {(it + 1)} Saved')

				# min_eval_loss = eval_loss

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

			loss_log = collections.defaultdict(float)
			st = ed

	save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
	net.cpu()
	torch.save({'state_dict': net.state_dict()}, save_pth)

	Logger.logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
	main()
