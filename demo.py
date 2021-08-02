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
from torch.utils.data import DataLoader
from model import model
from datasets import MelSpec2Audio
from hparams import create_hparams
from datasets.lrw import LRW
from datasets.wild import WILD
from datasets import train_collate_fn_pad, test_collate_fn_pad
from train_utils.tensorboard_logger import plot_spectrogram_to_numpy



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def demo(model_path):
	hparams = create_hparams()
	spec2audio = MelSpec2Audio(hparams, max_iters=256).to(device)

	net = model.get_network('test')
	net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)
	net.eval()

	net = net.to(device)

	ds = WILD('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/WILD', mode='test', duration=1, demo=True)
	# ds = LRW('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/LRW', mode='test', duration=1, demo=True)

	for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_collate_fn_pad):

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


		speaker_embedding = net.vgg_face.inference(face_crops[:, 0, :, :, :])
		
		mel_outputs, stop_tokens = net.inference(videos, face_crops, speaker_embedding)
		mel_outputs = mel_outputs[:1, :, :stop_tokens[0]]
		
		cv2.imshow('meloutput', plot_spectrogram_to_numpy(mel_outputs[0]))
		
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



		# # pred_text = perform_stt(speech, hparams.sampling_rate).lower()

		# # print(gt_text, pred_text, [True for p in pred_text.split(' ') if len(p) > 3 and p in gt_text], flush=True)
		# # input('Pred')

		# # if any([True for p in pred_text.split(' ') if len(p) > 3 and p in gt_text]):
		# # 	tp += 1
		# # total += 1		
		# # print(f'tp: {tp} total: {total} acc: {tp / total}')		

		# continue


		# file_path = file_paths[0]
		# save_path = file_path
		
		# path_split = file_path.split('/')
		# path_split[-2] = 'val'
		# path_split[-1] = path_split[-1][:-3] + 'wav'

		# path_split[-4] = 'GT'
		# os.makedirs('/'.join(path_split[:-1]), exist_ok=True)

		# video_path = save_path.split('/')
		# video_path[-4] = 'lipread_mp4'
		# video_path[-1] = video_path[-1][:-3] + 'mp4'

		# shutil.copy('/'.join(video_path), '/'.join(path_split)[:-3] + 'mp4')

		# sf.write('/'.join(path_split), gt_speech.astype(np.float32), hparams.sampling_rate)
		
		# path_split[-4] = 'PRED'
		# os.makedirs('/'.join(path_split[:-1]), exist_ok=True)
		
		# sf.write('/'.join(path_split), speech.astype(np.float32), hparams.sampling_rate)

		# print(path_split)
		# exit(0)
		# sf.write(filename, speech.astype(np.float32), hparams.sampling_rate)


def main():
	model_path = 'savedmodels/48a099a8cceee04f7e2d89774904a46f/368000_1627877370.pth'
	demo(model_path)


if __name__ == '__main__':
	main()





