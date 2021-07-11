import os
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from  hparams import create_hparams
try:
	from .modules import SpeakerEncoder, VideoExtractor, FaceRecognizer, Decoder
except ModuleNotFoundError: 
	from modules import SpeakerEncoder, VideoExtractor, FaceRecognizer, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoNameModel(nn.Module):
	def __init__(self, video_input_size=96, train=False):
		super().__init__()

		hparams = create_hparams()
		self._train_mode = train

		self.speaker_encoder = SpeakerEncoder()
		self.vgg_face = FaceRecognizer()

		self.v_encoder = VideoExtractor()
		# self.linear_projection = nn.Linear(256 + hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim))
		self.decoder = Decoder()


	def forward(self, video_frames, face_frames, audio_frames, melspecs, video_lengths, audio_lengths, melspec_lengths, tf_ratio):
		_, _, oldT, _, _ = video_frames.shape
		
		video_features = self.v_encoder(video_frames)
		
		N, T, C = video_features.shape

		encoder_lengths = video_lengths #torch.div(video_lengths, int(oldT / T), rounding_mode='trunc')
		
		face_features = self.vgg_face.inference(face_frames[:, 0, :, :, :])
		# face_features = self.speaker_encoder.inference(audio_frames)

		N, T, C = video_features.shape
		face_features = face_features.unsqueeze(1).repeat(1, T, 1)
		
		visual_features = torch.cat([video_features, face_features], dim=2)
		# visual_features = self.linear_projection(visual_features)
	
		outputs = self.decoder(visual_features, melspecs, encoder_lengths, melspec_lengths, tf_ratio)

		return outputs
				

	def inference(self, video_frames, face_frames):
		with torch.no_grad():
			video_features = self.v_encoder(video_frames)
						
			face_features = self.vgg_face(face_frames[:, 0, :, :, :])
			
			N, T, C = video_features.shape
			face_features = face_features.unsqueeze(1).repeat(1, T, 1)

			visual_features = torch.cat([video_features, face_features], dim=2)
			# visual_features = self.linear_projection(visual_features)

			outputs = self.decoder.inference(visual_features)

		return outputs


def get_network(mode):
	assert mode in ('train', 'test')

	model = NoNameModel(train=(mode == 'train'))

	if mode == 'train':
		model = model.train()
	else:
		model = model.eval()
	
	return model

def main():
	model = NoNameModel()

	video = torch.rand(4, 3, 25, 96, 96)
	face = torch.rand(4, 2, 3, 160, 160)
	speech = torch.rand(1, 16000 * 1)

	outputs = model.forward(video, face, audio_frames=0, melspecs=0, video_lengths=0, audio_lengths=0, melspec_lengths=0)

	print(outputs.shape)


if __name__ == '__main__':
	main()
