import os
from torch import Tensor
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

try:
	from hparams import create_hparams
except:
	import sys; sys.path.append('../..')
	from hparams import create_hparams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PSine(nn.Module):
	def __init__(self, dims, w=1, inplace=True):
		super().__init__()
		
		self.w = nn.Parameter(torch.ones(dims) * w)
		self.inplace = inplace
		
	def forward(self, x: Tensor):
		if self.inplace:
			return x.sin_().mul_(self.w)
		else:
			return torch.sin(x) * self.w


class LinearNorm(torch.nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
		super(LinearNorm, self).__init__()
		self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

		torch.nn.init.xavier_uniform_(
			self.linear_layer.weight,
			gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, x):
		return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
				 padding=None, dilation=1, bias=True, w_init_gain='linear'):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)

		self.conv = torch.nn.Conv1d(in_channels, out_channels,
									kernel_size=kernel_size, stride=stride,
									padding=padding, dilation=dilation,
									bias=bias)

		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal


class Postnet(nn.Module):
	"""Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	"""

	def __init__(self, hparams):
		super(Postnet, self).__init__()
		self.convolutions = nn.ModuleList()

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
						 kernel_size=hparams.postnet_kernel_size, stride=1,
						 padding=int((hparams.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='tanh'),
				nn.BatchNorm1d(hparams.postnet_embedding_dim))
		)

		for i in range(1, hparams.postnet_n_convolutions - 1):
			self.convolutions.append(
				nn.Sequential(
					ConvNorm(hparams.postnet_embedding_dim,
							 hparams.postnet_embedding_dim,
							 kernel_size=hparams.postnet_kernel_size, stride=1,
							 padding=int((hparams.postnet_kernel_size - 1) / 2),
							 dilation=1, w_init_gain='tanh'),
					nn.BatchNorm1d(hparams.postnet_embedding_dim))
			)

		self.sin_activation = nn.ModuleList([PSine(hparams.postnet_embedding_dim) for _ in range(0, hparams.postnet_n_convolutions - 1)])

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
						 kernel_size=hparams.postnet_kernel_size, stride=1,
						 padding=int((hparams.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='linear'),
				nn.BatchNorm1d(hparams.n_mel_channels))
			)

	def forward(self, x):
		for i in range(len(self.convolutions) - 1):
			x = self.convolutions[i](x).permute(0, 2, 1)
			x = self.sin_activation[i](x).permute(0, 2, 1)
			x = F.dropout(x, 0.5, self.training)
			
		x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

		return x


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()

		hparams = create_hparams()

		self.n_mel_channels = hparams.n_mel_channels
		self.postnet = Postnet(hparams)
		self.hparams = hparams

		FFN_HID_DIM = 512
		ENCODER_DIM = hparams.encoder_embedding_dim
		N_MELS = hparams.n_mel_channels

		self.initial_context = nn.Parameter(nn.Parameter(torch.randn(1, 1, N_MELS)))

		FFN_HID_DIM = 256
		self.encoder_rnn = nn.LSTM(ENCODER_DIM, FFN_HID_DIM, 2, bidirectional=True, batch_first=True)
		self.decoder_rnn = nn.LSTM(64, FFN_HID_DIM, 4, dropout=0.1, batch_first=True)
		self.fc_out = nn.Linear(FFN_HID_DIM, N_MELS)

		self.prenet = nn.Sequential(
			nn.Linear(N_MELS, 128),
			PSine(128),
			nn.Dropout(0.25),
			nn.Linear(128, 128),
			PSine(128),
			nn.Dropout(0.25),
			nn.Linear(128, 64),
		)
		self.stop_token_layer = LinearNorm(2 * FFN_HID_DIM, 1, w_init_gain='sigmoid')

		self.K = nn.Conv1d(512, 512, 9)			
		self.V = nn.Conv1d(512, 512, 9)				
		self.Q = nn.Conv1d(64, 512, 24, padding=24 // 2)
		self.projection = nn.Linear(512, 64)		
		self.gamma = nn.Parameter(torch.zeros(1))		

	def forward(self, encoder_outputs, mels, text_lengths, output_lengths, tf_ratio):
		mels = mels.permute(0, 2, 1)

		N, cur_max_step, C = mels.shape[:3]

		encoder_outputs, (hidden, cell) = self.encoder_rnn(encoder_outputs)

		k = self.K(encoder_outputs.permute(0, 2, 1))
		v = self.V(encoder_outputs.permute(0, 2, 1))
		a = torch.softmax(torch.bmm(k, v.permute(0, 2, 1)), dim=-1)

		encoder_forward_hidden = hidden[-2] 
		
		ys = torch.tile(self.initial_context, (N, 1, 1))

		teacher_input = torch.cat([ys, mels], dim=1)
		
		outputs = torch.zeros(N, cur_max_step, C).to(device)
		stop_tokens = torch.zeros(N, cur_max_step, 1).to(device)
		context = torch.zeros(N, cur_max_step, 64).to(device)
		for i in range(cur_max_step):

			if torch.rand(1) > tf_ratio:
				ys = teacher_input[:, i, :].unsqueeze(1)
			
			ys = self.prenet(ys)

			context[:, i, :] = ys[:, 0, :]
			q = self.Q(context[:, :i + 1, :].permute(0, 2, 1))
			o = torch.bmm(a, q).permute(0, 2, 1)[:, i, :]
			o = self.gamma * self.projection(o).unsqueeze(1)
					
			ys = ys + o 

			output, (hidden, cell) = self.decoder_rnn(ys,  (hidden, cell))
		   
			ys = self.fc_out(output)

			outputs[:, i, :] = ys.squeeze(1)

			stop_tokens[:, i, :] = self.stop_token_layer(torch.cat([hidden[-1], encoder_forward_hidden], dim=1))

		outputs = outputs.permute(0, 2, 1)    

		post_preds = self.postnet(outputs) + outputs

		return (outputs, post_preds, stop_tokens)
		
	
	def inference(self, encoder_outputs):
		N, src_seq_len = encoder_outputs.shape[:2]
				
		encoder_outputs, (hidden, cell) = self.encoder_rnn(encoder_outputs)
		k = self.K(encoder_outputs.permute(0, 2, 1))
		v = self.V(encoder_outputs.permute(0, 2, 1))
		a = torch.softmax(torch.bmm(k, v.permute(0, 2, 1)), dim=-1)
		
		encoder_forward_hidden = hidden[-2]
		
		ys = torch.tile(self.initial_context, (N, 1, 1))

		output_lengths = torch.ones(N, device=device, dtype=int) * self.hparams.max_decoder_steps
		outputs = torch.zeros(N, self.hparams.max_decoder_steps, self.hparams.n_mel_channels).to(device)
		context = torch.zeros(N, self.hparams.max_decoder_steps, 64).to(device)
		for i in range(self.hparams.max_decoder_steps):
			ys = self.prenet(ys)

			context[:, i, :] = ys[:, 0, :]
			q = self.Q(context[:, :i + 1, :].permute(0, 2, 1))
			o = torch.bmm(a, q).permute(0, 2, 1)[:, i, :]
			o = self.gamma * self.projection(o).unsqueeze(1)
				
			ys = ys + o 


			output, (hidden, cell) = self.decoder_rnn(ys,  (hidden, cell))

			stop_tokens = self.stop_token_layer(torch.cat([hidden[-1], encoder_forward_hidden], dim=1))
		
			stop_indices = (torch.sigmoid(stop_tokens) > 0.5).nonzero()
			
			if len(stop_indices):   
				for idx in stop_indices[:, 0]:
					if output_lengths[idx] == self.hparams.max_decoder_steps:
						output_lengths[idx] = i + 1
		

			ys = self.fc_out(output)

			outputs[:, i, :] = ys.squeeze(1)

		outputs = outputs.permute(0, 2, 1)    
		outputs = self.postnet(outputs) + outputs

		return outputs, output_lengths


def main():
	decoder = Decoder()

	visual_features, melspecs, encoder_lengths, melspec_lengths = torch.rand(36, 29, 1024), torch.rand(36, 80, 82), torch.ones(36, dtype=torch.long) * 29, torch.ones(36, dtype=torch.long) * 82
	outs = decoder(visual_features, melspecs, encoder_lengths, melspec_lengths)
	print(outs[0].shape)

	print(decoder.inference(torch.rand(36, 29, 1024)).shape)


if __name__ == "__main__":
	main()