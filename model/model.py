import os
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
try:
	from .modules import VideoExtractor, FaceRecognizer, Decoder
except ModuleNotFoundError: 
	from modules import VideoExtractor, FaceRecognizer, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SPILSS(nn.Module):
	def __init__(self):
		super().__init__()

		self.vgg_face = FaceRecognizer()

		self.v_encoder = VideoExtractor()
		self.decoder = Decoder()


	def forward(self, video_frames, face_frames, audio_frames, melspecs, video_lengths, audio_lengths, melspec_lengths, tf_ratio):
		_, _, oldT, _, _ = video_frames.shape
		
		video_features = F.dropout(self.v_encoder(video_frames), 0.3, self.training)
		encoder_lengths = video_lengths
		
		face_features = self.vgg_face.inference(face_frames[:, 0, :, :, :])
		N, T, C = video_features.shape
		face_features = face_features.unsqueeze(1).repeat(1, T, 1)
		
		visual_features = torch.cat([video_features, face_features], dim=2)
	
		outputs = self.decoder(visual_features, melspecs, encoder_lengths, melspec_lengths, tf_ratio)

		return outputs
				

	def inference(self, video_frames, face_frames, speaker_embedding=None, **kwargs):
		with torch.no_grad():
			video_features = self.v_encoder(video_frames)

			if speaker_embedding is None:			
				face_features = self.vgg_face.inference(face_frames[:, 0, :, :, :])
			else:
				face_features = speaker_embedding
				
			N, T, C = video_features.shape
			face_features = face_features.unsqueeze(1).repeat(1, T, 1)

			visual_features = torch.cat([video_features, face_features], dim=2)

			outputs = self.decoder.inference(visual_features, **kwargs)

		return outputs


def get_network(mode):
	assert mode in ('train', 'test')

	model = SPILSS()

	if mode == 'train':
		model = model.train()
	else:
		model = model.eval()
	
	return model

def main():
	model = SPILSS()

	video = torch.rand(4, 3, 25, 96, 96)
	face = torch.rand(4, 2, 3, 160, 160)
	speech = torch.rand(1, 16000 * 1)

	outputs = model.forward(video, face, audio_frames=0, melspecs=0, video_lengths=0, audio_lengths=0, melspec_lengths=0)

	print(outputs.shape)


if __name__ == '__main__':
	main()
