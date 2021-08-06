import torch
import torch.nn.functional as F
import torch.nn as nn
import random


try:
	from hparams import create_hparams
except:
	import sys; sys.path.append('../..')
	from hparams import create_hparams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ResidualBlock(nn.Module):
	def make_conv(self, in_chns, out_chns, kernel_size, stride=1, padding=1):
		return nn.Sequential(
			nn.Conv1d(in_chns, out_chns, kernel_size, stride, padding=padding),
			nn.BatchNorm1d(out_chns),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def __init__(self, in_chns, out_chns, k_size=3, down_sample=True):
		super().__init__()

		self.conv_1 = self.make_conv(in_chns, out_chns, k_size, padding=k_size//2)
		self.conv_2 = self.make_conv(out_chns, out_chns, k_size, 2 if down_sample else 1, padding=k_size//2)

		self.down_sample = nn.Conv1d(in_chns, out_chns, 1, 2 if down_sample else 1)

	def forward(self, x):
		residual = self.down_sample(x)

		x = self.conv_1(x)
		x = self.conv_2(x)

		return x + residual


class Discriminator(nn.Module):
	def __init__(self, speaker_embedding_dims=256):
		super().__init__()

		hparams  = create_hparams()
		in_chns = hparams.n_mel_channels

		self.mel_encoder = nn.Sequential(
			nn.Linear(in_chns, 256),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.speaker_encoder = nn.Sequential(
			nn.Linear(speaker_embedding_dims, 256),
			nn.Softsign(),
		)
		
		self.encoder = nn.Sequential(
			nn.Linear(512, 384),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(384, 256),
			nn.LeakyReLU(0.2, inplace=True),
			
			nn.Dropout(0.1),
		)

		self.decoder = nn.ModuleList([
			nn.Sequential(
				ResidualBlock(256, 256),
			),
			nn.Sequential(
				ResidualBlock(256, 384),
			),
			nn.Sequential(
				ResidualBlock(384, 512),
			),
			nn.Sequential(
				ResidualBlock(512, 768),
			),
			nn.Sequential(
				ResidualBlock(768, 1024, down_sample=False),	
			),
		])   

		self.fc = nn.Linear(1024, 1)

		self.rand_patch = None

	def generate_rand_patch(self, N, T):
		patch_size = random.choice(range(7, 32))
		start_idx = max(0, 1 + (torch.rand(1) * T).long() - patch_size)
		end_idx = start_idx + patch_size

		self.rand_patch = (start_idx, end_idx)

	def forward(self, x, speaker_embeddding, same_rand=False, return_features=False):
		N, C, T = x.shape

		if self.rand_patch is None: self.generate_rand_patch(N, T)
		if same_rand is False: self.generate_rand_patch(N, T)
		
		start_idx, end_idx = self.rand_patch
		x = x[:, :, start_idx: end_idx].permute(0, 2, 1)
				

		speaker_embeddding = speaker_embeddding.unsqueeze(1).repeat(1, x.shape[1], 1)
		
		x = torch.cat([self.mel_encoder(x), self.speaker_encoder(speaker_embeddding)], dim=-1)

		x = self.encoder(x).permute(0, 2, 1)
		

		features = list()
		for layer in self.decoder:
			x = layer(x)
			features.append(x)
		
		x = F.adaptive_avg_pool1d(x, 1).view(N, -1)
		
		output = self.fc(F.dropout(x, 0.2, self.training)).view(N)

		if return_features:
			return output, features

		return output 


def main():
	from torchstat import stat

	model = Discriminator()
	model.eval()
	
	inp = torch.rand(5, 80 + 256, 77)

	# for i in range(100):
	outputs = model(torch.rand(5, 80, 77), torch.rand(5, 256))
	# print(outputs[:, 0, :8])
	print(outputs.shape)
	
	# outputs = model(inp[:, :, :6, :, :])
	# print(outputs[:, 0, :8])

	# print(outputs.shape)


if __name__ == '__main__':
	main()
