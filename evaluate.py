#!/usr/bin/env python3

import torch
import os
from datasets import MelSpec2Audio
from hparams import create_hparams
from torch.utils.data import DataLoader
from model import model
from datasets.lrw import LRW
from datasets.grid import GRID
from datasets.avspeech import AVSpeech 
from datasets.wild import WILD
from datasets import train_collate_fn_pad
from pystoi import stoi

import arg_parser 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_net(net, val_dataset):
	hparams = create_hparams()
	spec2audio = MelSpec2Audio(hparams, max_iters=256).to(device)
	
	net.eval()

	BATCH_SIZE = 32


	ESTOI = list()
	for batch in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), collate_fn=train_collate_fn_pad):
		(videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates), face_crops = batch

		gt_audio = audios.numpy()

		with torch.no_grad(): 
			mel_outputs = net(videos.to(device), face_crops.to(device), audios.to(device), melspecs.to(device), video_lengths.to(device), audio_lengths.to(device), melspec_lengths.to(device), 1)[1]
		

		pred_audio = spec2audio(mel_outputs).cpu().numpy()
		
		
		for i in range(gt_audio.shape[0]):
			ESTOI.append(stoi(gt_audio[i], pred_audio[i], hparams.sampling_rate, extended=True))
		
	mean_ESTOI = sum(ESTOI) / len(ESTOI)
		
	net.train()	

	return mean_ESTOI


def main():
	args = arg_parser.evaluate()

	model_path = args.saved_model

	dataset_name = args.dataset
	dataset_path = args.dataset_path
	
	if dataset_name == 'LRW':
		ds = LRW(dataset_path, mode='test', duration=1)
	elif dataset_name == 'GRID':
		ds = GRID(dataset_path, mode='test', duration=1)
	elif dataset_name == 'AVSpeech':
		ds = AVSpeech(dataset_path, mode='test', duration=1)
	elif dataset_name == 'WILD':
		ds = WILD(dataset_path, mode='test', duration=1)
	else: 
		raise FileNotFoundError("Dataset Not Present")

	state_dict = torch.load(model_path, map_location=device)
	if 'state_dict' in state_dict: state_dict = state_dict['state_dict']

	net = model.get_network('test')
	net.load_state_dict(state_dict, strict=False)
	net = net.to(device)

	mean_ESTOI = evaluate_net(net, ds)
	print(f"ESTOI for {dataset_name}: {mean_ESTOI}")


if __name__ == '__main__':
	main()
