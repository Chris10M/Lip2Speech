import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from model import model
from model.modules import SpeakerEncoder
from datasets import MelSpec2Audio
from hparams import create_hparams
from datasets.grid import GRID
from datasets.avspeech import AVSpeech
from datasets.lrw import LRW
from datasets.wild import WILD
from datasets import train_collate_fn_pad, test_collate_fn_pad
from train_utils.tensorboard_logger import plot_spectrogram_to_numpy, plot_alignment_to_numpy

import arg_parser 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(args):
	hparams = create_hparams()
	spec2audio = MelSpec2Audio(hparams, max_iters=256).to(device)

	net = model.get_network('test')
	
	state_dict = torch.load(args.saved_model, map_location=device)
	if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
	
	speaker_encoding_State_dict = dict()
	for k in list(state_dict.keys()):
		if k.startswith('speaker_encoder.'):
			speaker_encoding_State_dict[k[len('speaker_encoder.'):]] = state_dict.pop(k)
	
	net.load_state_dict(state_dict, strict=True)
	net.eval()

	net = net.to(device)

	speaker_encoder = SpeakerEncoder(state_dict=speaker_encoding_State_dict).eval().to(device)

	dataset_name = args.dataset
	dataset_path = args.dataset_path
	
	if dataset_name == 'LRW':
		ds = LRW(dataset_path, mode='test', duration=1, demo=True)
	elif dataset_name == 'GRID':
		ds = GRID(dataset_path, mode='test', duration=1, demo=True)
	elif dataset_name == 'AVSpeech':
		ds = AVSpeech(dataset_path, mode='test', duration=1, demo=True)
	elif dataset_name == 'WILD':
		ds = WILD(dataset_path, mode='test', duration=1, demo=True)
	else:
		raise FileNotFoundError("Dataset Not Present")


	for bdx, batch in enumerate(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_collate_fn_pad)):
		(videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates), face_crops, file_paths = batch
		
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

		with torch.no_grad():
			if args.encoding == 'face':
				speaker_embedding = net.vgg_face.inference(face_crops[:, 0, :, :, :])
			elif args.encoding == 'voice':
				speaker_embedding = speaker_encoder.inference(audios.to(device))
			
			mel_outputs, stop_tokens, attention_matrix = net.inference(videos.to(device), face_crops.to(device), speaker_embedding.to(device), return_attention_map=True)
			mel_outputs = mel_outputs[:1, :, :stop_tokens[0]]
	
		gt_speech = spec2audio(melspecs.to(device)).cpu()[0].numpy()
		pred_speech = spec2audio(mel_outputs).cpu()[0].numpy()

		stop_tokens = stop_tokens.cpu()
		mel_outputs = mel_outputs.cpu()
		attention_matrix = attention_matrix.cpu()

		cv2.imshow('attention', plot_alignment_to_numpy(attention_matrix[0, :stop_tokens[0], :].T))
		cv2.imshow('meloutput', plot_spectrogram_to_numpy(mel_outputs[0]))
		cv2.imshow('melgt', plot_spectrogram_to_numpy(melspecs[0]))
		
		sd.stop()
		sd.play(gt_speech, hparams.sampling_rate)
		
		print('Ground Truth Speech')
		
		if ord('q') == cv2.waitKey(1500):
			exit()
		
		
		speech = np.pad(pred_speech, (0, hparams.sampling_rate), mode="constant")
		
		sd.stop()
		sd.play(speech, hparams.sampling_rate)

		sf.write('gt.wav', gt_speech.astype(np.float32), hparams.sampling_rate)
		sf.write('pred.wav', speech.astype(np.float32), hparams.sampling_rate)

		print('Predicted Speech')
		
		if ord('q') == cv2.waitKey(1500):
			exit()


def main():
	args = arg_parser.demo()
	demo(args)


if __name__ == '__main__':
	main()





