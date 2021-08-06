import shutil
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import os
import speech_recognition as sr
import imutils
import time
from torch._C import dtype
from torch.utils.data import DataLoader
from model import model
from datasets import MelSpec2Audio
from hparams import create_hparams
from datasets.lrw import LRW
from datasets.wild import WILD
from datasets import train_collate_fn_pad, test_collate_fn_pad
from train_utils.tensorboard_logger import plot_spectrogram_to_numpy, plot_alignment_to_numpy



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def demo(model_path):
	hparams = create_hparams()
	spec2audio = MelSpec2Audio(hparams, max_iters=256).to(device)

	net = model.get_network('test')
	net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)
	net.eval()

	net = net.to(device)

	# ds = WILD('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/WILD', mode='test', duration=1, demo=True)
	ds = LRW('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/LRW', mode='test', duration=1, demo=True)

	for batch in DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=test_collate_fn_pad):

		(videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates), face_crops, file_paths = batch
		speech = audios[:1].cpu()


		face = face_crops[0, 0, :, :, :].permute(1, 2, 0).numpy()
		face = ((face * 128.0) + 127.5).astype(dtype=np.uint8)
		face = face[:, :, :: -1]

		cv2.imshow('face', face)
		
		B, C, T, H, W = videos.shape
		for i in range(T):
			image = videos[0, :, i, :, :].permute(1, 2, 0).numpy()
			image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
			image = image[:, :, :: -1].astype(dtype=np.uint8)
	
			cv2.imshow('mouthROI', image)
			
			if ord('q') == cv2.waitKey(25):
				exit()

		# speaker_embedding  = torch.from_numpy(np.load('speaker_embedding.npy'))
		speaker_embedding = net.vgg_face.inference(face_crops[:, 0, :, :, :])
		
		mel_outputs, stop_tokens, a = net.inference(videos, face_crops, speaker_embedding, return_attention_map=True)
		# attention_matrix = a

		# attention_gt = torch.ones_like(attention_matrix) * - 1
		# for bdx in range(0, stop_tokens.shape[0]):
		# 	seq_len = (stop_tokens[bdx] == 1).nonzero(as_tuple=True)[0] + 1
			
		# 	for i in range(seq_len):
		# 		attention_gt[bdx, i] = 0
		# 		adx = int((i / seq_len) * attention_gt.shape[2])
		# 		attention_gt[bdx, i, adx] = 1       

		# print(attention_matrix.shape)
		mel_outputs = mel_outputs[:1, :, :stop_tokens[0]]
		
		melspecs = torch.trunc(melspecs)
		print(torch.unique(melspecs))
		# attention = a[0, :, :].numpy() * 255
		
		cv2.imshow('attention', plot_alignment_to_numpy(a[0, :stop_tokens[0], :].numpy().T))
		cv2.imshow('meloutput', plot_spectrogram_to_numpy(mel_outputs[0]))
		cv2.imshow('melgt', plot_spectrogram_to_numpy(melspecs[0]))

		gt_speech = spec2audio(melspecs).cpu()[0].numpy()
		sd.stop()
		sd.play(gt_speech, hparams.sampling_rate)
		
		print('Ground Truth Speech')
		
		if ord('q') == cv2.waitKey(1500):
			exit()
		
		speech = spec2audio(mel_outputs).cpu()[0].numpy()
		speech = np.pad(speech, (0, hparams.sampling_rate), mode="constant")
		
		sd.stop()
		sd.play(speech, hparams.sampling_rate)

		print('Predicted Speech')
		
		if ord('q') == cv2.waitKey(1500):
			exit()


def main():
	model_path = 'savedmodels/48a099a8cceee04f7e2d89774904a46f/574000_1628185435.pth'
	demo(model_path)


if __name__ == '__main__':
	main()





