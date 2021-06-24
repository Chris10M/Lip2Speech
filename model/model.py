import os
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import AudioExtractor, VideoExtractor, FaceRecognizer, Decoder
except ModuleNotFoundError: 
    from modules import AudioExtractor, VideoExtractor, FaceRecognizer, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoNameModel(nn.Module):
    def __init__(self, video_input_size=96, train=False):
        super().__init__()

        self.vgg_face = FaceRecognizer()
        self.video_fe = VideoExtractor(video_input_size)

        self.decoder = Decoder()
        hparams = self.decoder.hparams
        
        self._train_mode = train

        self.lstm = nn.LSTM(1024, # FaceRecognizer embedding Size
		                    int(hparams.encoder_embedding_dim / 2), 1,
		                    batch_first=True, bidirectional=True)


    def forward(self, video_frames, face_frames, audio_frames, melspecs, video_lengths, audio_lengths, melspec_lengths):
        _, _, oldT, _, _ = video_frames.shape
        
        video_features = self.video_fe(video_frames)
        
        N, T, C = video_features.shape
        encoder_lengths = video_lengths // int(oldT / T)
        
        face_pair_1 = self.vgg_face(face_frames[:, 0, :, :, :])
        # face_pair_2 = self.vgg_face(face_frames[:, 1, :, :, :])
        
        face_features = face_pair_1.unsqueeze(0).repeat(2, 1, 1)
        h_0 = torch.zeros_like(face_features).to(face_features.device)
        visual_features, _ = self.lstm(video_features, (h_0, face_features)) 
        
        # print(melspecs.permute(0, 2, 1), visual_features.shape, melspecs.permute(0, 2, 1).shape)
        # exit(0)
        
        outputs = self.decoder(visual_features, melspecs, encoder_lengths, melspec_lengths)

        return outputs, (None, None)
        # return outputs, (face_pair_1, face_pair_2)

    def inference(self, video_frames, face_frames):
        with torch.no_grad():
            video_features = self.video_fe(video_frames)
                        
            face_features = self.vgg_face(face_frames[:, 0, :, :, :])
            
            face_features = face_features.unsqueeze(0).repeat(2, 1, 1)
            h_0 = torch.zeros_like(face_features).to(face_features.device)
            visual_features, _ = self.lstm(video_features, (h_0, face_features))

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
    face = torch.rand(4, 1, 3, 160, 160)
    speech = torch.rand(1, 16000 * 1)

    outputs = model.forward(video, face, audio_frames=0, melspecs=0, video_lengths=0, audio_lengths=0, melspec_lengths=0)

    print(outputs.shape)


if __name__ == '__main__':
    main()
