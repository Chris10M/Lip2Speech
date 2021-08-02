import torch
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
import math
import numpy as np


try:
	from shufflenetv2 import ShuffleNetV2
except:
	from .shufflenetv2 import ShuffleNetV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -- auxiliary functions
def threeD_to_2D_tensor(x):
	n_batch, n_channels, s_time, sx, sy = x.shape
	x = x.transpose(1, 2)
	return x.reshape(n_batch*s_time, n_channels, sx, sy)


class VideoExtractor(nn.Module):
	def _initialize_weights_randomly(self):

		use_sqrt = True

		if use_sqrt:
			def f(n):
				return math.sqrt( 2.0/float(n) )
		else:
			def f(n):
				return 2.0/float(n)

		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
				n = np.prod( m.kernel_size ) * m.out_channels
				m.weight.data.normal_(0, f(n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			elif isinstance(m, nn.Linear):
				n = float(m.weight.data[0].nelement())
				m.weight.data = m.weight.data.normal_(0, f(n))

	def __init__( self, modality='video', hidden_dim=256, backbone_type='shufflenet', num_classes=500,
				  relu_type='prelu', tcn_options={}, width_mult=1.0, extract_feats=False):
		super(VideoExtractor, self).__init__()
		self.extract_feats = extract_feats
		self.backbone_type = backbone_type
		self.modality = modality

		assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
		
		shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
		self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
		self.frontend_nout = 24
		self.backend_out = shufflenet.stage_out_channels[-1]

		frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
		self.frontend3D = nn.Sequential(
					nn.Conv3d(3, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
					nn.BatchNorm3d(self.frontend_nout),
					frontend_relu,
					nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

		self._initialize_weights_randomly()

	def forward(self, x):
		B, C, T, H, W = x.size()
		x = self.frontend3D(x)
		Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
		x = threeD_to_2D_tensor( x )
		x = self.trunk(x)
		x = x.view(-1, self.backend_out)
		x = x.view(B, Tnew, x.size(1))
		
		x = F.normalize(x, p=2, dim=2)
		
		return x


def main():
	from torchstat import stat

	model = VideoExtractor()
	model.eval()
	# model.load_state_dict(torch.load('/media/ssd/christen-rnd/Experiments/Lip2Speech/lrw_snv1x_dsmstcn3x.pth.tar')['model_state_dict'])
	
	inp = torch.rand(5, 3, 30, 96, 96)

	outputs = model(inp)
	# print(outputs[:, 0, :8])
	print(outputs.shape)
	
	# outputs = model(inp[:, :, :6, :, :])
	# print(outputs[:, 0, :8])

	# print(outputs.shape)


if __name__ == '__main__':
	main()
