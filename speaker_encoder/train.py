import torch
from torch import nn
import numpy as np
torch.manual_seed(1)
from logger import setup_logger
import os
import traceback
from torch import optim as Optimzer
import collections
from torch.utils.data import DataLoader, dataset
import hashlib
import os
import os.path as osp
import logging
import time
import datetime
from tensorboard_logger import TFBoard

from losses import *
from dataset import AVSpeechFace, av_speech_face_collate_fn
from model import get_network
from model import *
# from evaluate import evaluate_net


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
	Logger.tensor_board = TFBoard(tf_logs)


def main():	
	ds = AVSpeechFace('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='train')

	face_decoder = FaceDecoder()
	face_decoder = face_decoder.train().to(device)
	net, speech_encoder = get_network()
	set_model_logger(net)
	
	saved_path = 'savedmodels/156bcbfe7c66281240affb1d053dd279/438000_1625379312.pth'
	
	max_iter = 720000
	save_iter = 1000
	n_img_per_gpu = 64
	n_workers = 16# min(n_img_per_gpu, os.cpu_count())
 
	dl = DataLoader(ds,
					batch_size=n_img_per_gpu,
					shuffle=True,
					num_workers=n_workers,
					pin_memory=True,
					drop_last=False, 
					collate_fn=av_speech_face_collate_fn, 
					persistent_workers=True,
					prefetch_factor=4)

	contrastive_criterion = MiniBatchConstrastiveLoss()
	reconstuction_criterion = ReconstuctionLoss()

	optim = Optimzer.SGD(net.parameters(), weight_decay=1e-5, lr=1e-3, momentum=0.9)
	t_optim = Optimzer.Adam([contrastive_criterion.t])
	f_optim = Optimzer.Adam(face_decoder.parameters())
	
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

		optim.load_state_dict(loaded_model['optimize_state'])

		if 't' in loaded_model:
			t_optim.load_state_dict(loaded_model['t']['optim'])
			contrastive_criterion.t = loaded_model['t']['value']

		if 'face_decoder' in loaded_model:
			f_optim.load_state_dict(loaded_model['face_decoder']['optim'])
			face_decoder.load_state_dict(loaded_model['face_decoder']['state_dict'])

		print(f'Model Loaded: {saved_path} @ start_it: {start_it} t: {contrastive_criterion.t}')

	for group in optim.param_groups: group['initial_lr'] = 1e-3
	scheduler = Optimzer.lr_scheduler.CosineAnnealingLR(optim, (max_iter * n_img_per_gpu) // len(ds), last_epoch=(start_it * n_img_per_gpu)// len(ds), verbose=True)

	## train loop
	msg_iter = 50
	loss_logs = collections.defaultdict(float)
	st = glob_st = time.time()
	diter = iter(dl)
	epoch = 0

	net = net.train()
	for it in range(start_it, max_iter):
		try:
			batch = next(diter)
		except StopIteration:
			diter = iter(dl)
			epoch += 1
			scheduler.step()
			
			batch = next(diter)

		speeches, faces = batch
		speeches, faces = speeches.to(device), faces.to(device)

		try:		
			with torch.no_grad():
				speech_embeddings = speech_encoder(speeches)

			face_embeddings = net(faces)

			if torch.rand(1) > 0.5: 
				reconstucted_faces = face_decoder(face_embeddings.detach())
			else:
				reconstucted_faces = face_decoder(speech_embeddings)
		except: 
			traceback.print_exc()
			continue
		
		losses = dict()
		losses = contrastive_criterion([speech_embeddings, face_embeddings], losses)
		losses = reconstuction_criterion(reconstucted_faces, faces, losses)

		loss = list()
		for k, v in losses.items():
			loss_logs[k] += v.item()
			loss.append(v)
		
		loss = sum(loss)

		optim.zero_grad()
		t_optim.zero_grad()
		f_optim.zero_grad()
		
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
		
		optim.step()
		t_optim.step()
		f_optim.step()
	
		loss_logs['loss'] += loss.item()
		

		if (it + 1) % save_iter == 0:
				save_pth = osp.join(Logger.ModelSavePath, f'{it + 1}_{int(time.time())}.pth')

			# evaluation = evaluate_net(args, net)
			# Logger.logger.info(f"Model@{it + 1}\n{evaluation}"
			# optim.reduce_lr_on_plateau(eval_loss)
				eval_loss = loss
				# Logger.tensor_board.log_validation(eval_loss, net, (melspecs, mel_gates), outputs, it + 1)

			# if eval_loss < min_eval_loss:  
				print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
				torch.save({
					'start_it': it,
					'state_dict': net.state_dict(),
					'optimize_state': optim.state_dict(),
					'min_eval_loss': min_eval_loss,
					't': {'value': contrastive_criterion.t, 'optim': t_optim.state_dict()},
					'face_decoder': {'state_dict': face_decoder.state_dict(),  'optim': f_optim.state_dict()},
				}, save_pth)
				print(f'model at: {(it + 1)} Saved')

				# min_eval_loss = eval_loss

		#   print training log message
		if (it+1) % msg_iter == 0:
			for k, v in loss_logs.items(): loss_logs[k] = round(v / msg_iter, 6)

			ed = time.time()
			t_intv, glob_t_intv = ed - st, ed - glob_st
			eta = int((max_iter - it) * (glob_t_intv / it))
			eta = str(datetime.timedelta(seconds=eta))
			msg = ', '.join([
					f'epoch: {epoch}',
					'it: {it}/{max_it}',         
					*[f"{k}: {v}" for k, v in loss_logs.items()],
					'eta: {eta}',
					'time: {time:.2f}',
				]).format(
					it = it+1,
					max_it = max_iter,
					time = t_intv,
					eta = eta
				)
				
			Logger.logger.info(msg)
			Logger.tensor_board.log_training(loss_logs['loss'], grad_norm, 1, t_intv, it + 1)
			
			# Logger.tensor_board.log_alignment(alignments, it + 1)

			loss_logs = collections.defaultdict(float)
			st = ed

	save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
	net.cpu()
	torch.save({'state_dict': net.state_dict()}, save_pth)

	Logger.logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
	main()
