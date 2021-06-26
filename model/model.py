import os
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

try:
	from .modules import SpeakerEncoder, VideoExtractor, FaceRecognizer, Decoder
except ModuleNotFoundError: 
	from modules import SpeakerEncoder, VideoExtractor, FaceRecognizer, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoNameModel(nn.Module):
	def __init__(self, video_input_size=96, train=False):
		super().__init__()

		self.vgg_face = FaceRecognizer()
		self.video_fe = VideoExtractor(video_input_size)

		self.decoder = Decoder()
		hparams = self.decoder.hparams
		
		self._train_mode = train

		self.visual_lstm = nn.LSTM(512 + hparams.encoder_embedding_dim, 
								   int(hparams.encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True)

		# self.pre_lstm = nn.LSTM(513, int(hparams.encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True)

		# if train:
		#     self.spec_encoder = SpecEncoder()


	def forward(self, video_frames, face_frames, audio_frames, melspecs, video_lengths, audio_lengths, melspec_lengths):
		_, _, oldT, _, _ = video_frames.shape
		
		video_features = self.video_fe(video_frames)
		
		N, T, C = video_features.shape
		encoder_lengths = video_lengths // int(oldT / T)
		
		face_pair_1 = self.vgg_face(face_frames[:, 0, :, :, :])
		# face_pair_2 = self.vgg_face(face_frames[:, 1, :, :, :])
		
		face_features = face_pair_1
		N, T, C = video_features.shape
		face_features = face_features.unsqueeze(1).repeat(1, T, 1)
		
		visual_features = F.dropout(torch.cat([face_features, video_features], dim=2), 0.5, self.training)
		visual_features, _ = self.visual_lstm(visual_features)
		
		# visual_features, _ = self.pre_lstm(melspecs.permute(0, 2, 1))
		# encoder_lengths = melspec_lengths


		outputs = self.decoder(visual_features, melspecs, encoder_lengths, melspec_lengths)


		mel_outputs, _, _, _ = outputs
		# encoded_mel_gt =  self.spec_encoder(melspecs)
		# encoded_mel_pred = self.spec_encoder(mel_outputs)


		return outputs, (None, None, None, None)
		
		# return outputs, (face_pair_1, face_pair_2, encoded_mel_gt, encoded_mel_pred)
		

	def inference(self, video_frames, face_frames):
		with torch.no_grad():
			video_features = self.video_fe(video_frames)
						
			face_features = self.vgg_face(face_frames[:, 0, :, :, :])
			
			N, T, C = video_features.shape
			face_features = face_features.unsqueeze(1).repeat(1, T, 1)
			visual_features, _ = self.visual_lstm(torch.cat([face_features, video_features], dim=2))

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
