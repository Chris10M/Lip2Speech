from operator import le
import torch
import os
from datasets import MelSpec2Audio
from hparams import create_hparams
from torch.utils.data import DataLoader
from model import model
from datasets.lrw import LRW
from datasets.wild import WILD
from datasets import train_collate_fn_pad
from pystoi import stoi


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
	# model_path = 'savedmodels/d915e48826cab344bc1535e895d5959e/262000_1626274277.pth'
	# model_path = 'savedmodels/d915e48826cab344bc1535e895d5959e/240000_1626110322.pth'
	# model_path = 'savedmodels/d915e48826cab344bc1535e895d5959e/234000_1626101955.pth'
	# model_path = 'single_speaker_savedmodels/Elon_Single/f2096ba67a7bae647dec1b5c0a3f055e/356000_1626579116.pth'
	# model_path = 'savedmodels/f2096ba67a7bae647dec1b5c0a3f055e/334000_1626678346.pth'
	# model_path = 'savedmodels/f2096ba67a7bae647dec1b5c0a3f055e/392000_1626791363.pth'
	# model_path = 'savedmodels/2fa10a6f0f2b04f8a963b0205bbdd4a5/182000_1627100214.pth'
	model_path = 'savedmodels/48a099a8cceee04f7e2d89774904a46f/368000_1627877370.pth'

	net = model.get_network('test')
	net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)
	net = net.to(device)

	ds = LRW('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/LRW', mode='test', duration=1)
	evaluate_net(net, ds)


if __name__ == '__main__':
	main()
