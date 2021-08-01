import os
from torch import Tensor
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

try:
	from hparams import create_hparams
except:
	import sys; sys.path.append('../..')
	from hparams import create_hparams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


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


class MultiHopConv(nn.Module):
	def __init__(self, in_chns, out_chns):
		super().__init__()

		self.conv = nn.ModuleList([
			nn.Conv1d(in_chns, in_chns, 1),
			nn.Conv1d(in_chns, in_chns, 3, padding=3//2),
			nn.Conv1d(in_chns, in_chns, 7, padding=7//2),
			nn.Conv1d(in_chns, in_chns, 11, padding=11//2),
			nn.Conv1d(in_chns, in_chns, 15, padding=15//2),
		])

		self.bottleneck = nn.Conv1d(512 * len(self.conv), out_chns, 1)

	def forward(self, x):
		features = list()
		for layer in self.conv:
			features.append(layer(x))
		
		x = torch.cat(features, dim=1)
		x = self.bottleneck(x)
		
		return x


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()

		hparams = create_hparams()

		self.n_mel_channels = hparams.n_mel_channels
		self.postnet = Postnet(hparams)
		self.hparams = hparams

		ENCODER_DIM = hparams.encoder_embedding_dim
		N_MELS = hparams.n_mel_channels
		FFN_HID_DIM = 256
		
		self.initial_context = nn.Parameter(torch.randn(1, 1, N_MELS))

		self.encoder_rnn = nn.LSTM(ENCODER_DIM, FFN_HID_DIM, 2, dropout=0.1, bidirectional=True, batch_first=True)
		self.decoder_rnn = nn.LSTM(64, FFN_HID_DIM, 4, dropout=0.1, batch_first=True)
		self.fc_out = LinearNorm(FFN_HID_DIM, N_MELS)  

		self.prenet = nn.Sequential(
			LinearNorm(N_MELS, 128),
			PSine(128),
			nn.Dropout(0.2),
			LinearNorm(128, 128),
			PSine(128),
			nn.Dropout(0.2),
			LinearNorm(128, 64),
		)
		self.stop_token_layer = LinearNorm(2 * FFN_HID_DIM, 1, w_init_gain='sigmoid')

		self.K = MultiHopConv(512, 512) 			
		self.V = MultiHopConv(512, 512) 		
		self.Q = LinearNorm(64, 512)
		self.H_Q = LinearNorm(1024, 512)
		self.temperature = nn.Parameter(torch.ones(1) * (512 ** 0.5))
		
		self.attention_proj = LinearNorm(512, 64)
		self.layer_norm = nn.LayerNorm(64)
		
		self.encoder_site = nn.Sequential(LinearNorm(256, 256), nn.Softsign())
		self.attention_site = nn.Sequential(LinearNorm(256, 512), nn.Softsign())
		self.residual_bottleneck = nn.Conv1d(1024, 512, 1)

		self.positional_encodings = PositionalEncoding(1024, n_position=hparams.max_decoder_steps)

	def forward(self, encoder_outputs, face_features, mels, text_lengths, output_lengths, tf_ratio):
		residual = self.residual_bottleneck(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)

		mels = mels.permute(0, 2, 1)
		face_features = face_features[:, 0]

		N, cur_max_step, C = mels.shape[:3]

		encoder_site_embeddings = self.encoder_site(face_features).unsqueeze(0).repeat(4, 1, 1)
		attention_site_embeddings = self.attention_site(face_features).unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)

		encoder_outputs, (hidden, cell) = self.encoder_rnn(encoder_outputs, (encoder_site_embeddings, encoder_site_embeddings)) 
		encoder_outputs = F.softsign(encoder_outputs) + attention_site_embeddings + residual

		k = self.K(encoder_outputs.permute(0, 2, 1)) + self.positional_encodings(encoder_outputs)[:, :, :512].permute(0, 2, 1)
		v = self.V(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1) + self.positional_encodings(encoder_outputs)[:, :, :512]
		
		encoder_forward_hidden = hidden[-2] 
		
		ys = torch.tile(self.initial_context, (N, 1, 1))

		teacher_input = torch.cat([ys, mels], dim=1)
		teacher_consumed = 0
		
		outputs = torch.zeros(N, cur_max_step, C).to(device)
		stop_tokens = torch.zeros(N, cur_max_step, 1).to(device)
		positional_encodings = self.positional_encodings(outputs)
		for i in range(cur_max_step):
			
			if torch.rand(1) > tf_ratio and teacher_consumed < int(tf_ratio * cur_max_step):
				teacher_consumed += 1
				ys = teacher_input[:, i, :].unsqueeze(1)
				
			ys = self.prenet(ys)
			
			q = self.Q(F.softsign(ys)) + self.H_Q(F.softsign(torch.cat([h for h in hidden], dim=1))).unsqueeze(1) + positional_encodings[:, i, :512] 
			
			a = torch.softmax(torch.bmm(q * self.temperature, k), dim=-1)
			a = F.dropout(a, 0.1, self.training)
			o = self.attention_proj(torch.bmm(a, v))
					
			ys = self.layer_norm(ys + o)
			
			output, (hidden, cell) = self.decoder_rnn(ys,  (hidden, cell))
		   
			ys = self.fc_out(output)

			outputs[:, i, :] = ys.squeeze(1)

			stop_tokens[:, i, :] = self.stop_token_layer(torch.cat([hidden[-1], encoder_forward_hidden], dim=1))
		
		outputs = outputs.permute(0, 2, 1)    

		post_preds = self.postnet(outputs) + outputs

		return (outputs, post_preds, stop_tokens, face_features)
		
	
	def inference(self, encoder_outputs, face_features, return_attention_map=False):
		residual = self.residual_bottleneck(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
		
		N, src_seq_len = encoder_outputs.shape[:2]
		face_features = face_features[:, 0]

		encoder_site_embeddings = self.encoder_site(face_features).unsqueeze(0).repeat(4, 1, 1)
		attention_site_embeddings = self.attention_site(face_features).unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)

		encoder_outputs, (hidden, cell) = self.encoder_rnn(encoder_outputs, (encoder_site_embeddings, encoder_site_embeddings)) 
		encoder_outputs = F.softsign(encoder_outputs) + attention_site_embeddings + residual


		k = self.K(encoder_outputs.permute(0, 2, 1)) + self.positional_encodings(encoder_outputs)[:, :, :512].permute(0, 2, 1)
		v = self.V(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1) + self.positional_encodings(encoder_outputs)[:, :, :512]
		

		encoder_forward_hidden = hidden[-2]
		
		ys = torch.tile(self.initial_context, (N, 1, 1))

		output_lengths = torch.ones(N, device=device, dtype=int) * self.hparams.max_decoder_steps
		outputs = torch.zeros(N, self.hparams.max_decoder_steps, self.hparams.n_mel_channels, device=device, dtype=ys.dtype)
		positional_encodings = self.positional_encodings(outputs)

		attention_matrix = list()
		for i in range(self.hparams.max_decoder_steps):
			ys = self.prenet(ys)

			q = self.Q(F.softsign(ys)) + self.H_Q(F.softsign(torch.cat([h for h in hidden], dim=1))).unsqueeze(1) + positional_encodings[:, i, :512] 
			
			a = torch.softmax(torch.bmm(q * self.temperature, k), dim=-1)
			o = self.attention_proj(torch.bmm(a, v))
			attention_matrix.append(a)
					
			ys = self.layer_norm(ys + o)
			
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

		if return_attention_map:
			return outputs, output_lengths, torch.cat(attention_matrix, dim=1)

		return outputs, output_lengths


def main():
	import time
	t = time.time()
	decoder = Decoder()

	visual_features, face_features, melspecs, encoder_lengths, melspec_lengths = torch.rand(36, 29, 1024), torch.rand(36, 29, 256), torch.rand(36, 80, 82), torch.ones(36, dtype=torch.long) * 29, torch.ones(36, dtype=torch.long) * 82
	outs = decoder(visual_features, face_features, melspecs, encoder_lengths, melspec_lengths, 0.5)
	print(outs[0].shape, time.time() - t)

	# print(decoder.inference(torch.rand(36, 29, 1024)).shape)


if __name__ == "__main__":
	main()