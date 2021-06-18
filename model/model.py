import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import AudioExtractor, VideoExtractor, VGG_16
except ModuleNotFoundError: 
    from modules import AudioExtractor, VideoExtractor, VGG_16


class NoNameModel(nn.Module):
    def __init__(self, face_reconizer_path, video_input_size=96, train=False, wav2vec_path=''):
        super().__init__()

        self.vgg_face = VGG_16(face_reconizer_path)
        self.vgg_face.freeze_backbone()

        self.video_fe = VideoExtractor(video_input_size)

        self._train_mode = train

        if train:   
            assert os.path.isfile(wav2vec_path)

            self.wav2vec = AudioExtractor(wav2vec_path)

        self.video_pool = torch.nn.AvgPool2d((2, 1), stride=(2, 1))

    def forward(self, video_frames, face_frames, audio_frames):
        video_features = self.video_fe(video_frames)
        video_features = self.video_pool(video_features) # Assuming each lip-movement takes at least 2 frame. Intution: reduce frames to help LSTM to capture properly(longer dependacy issue) and also improve compute performance. 

        face_features = self.vgg_face(face_frames)

        if self._train_mode:
            audio_identity_features = self.wav2vec.identity_features(audio_frames)
        
        N, T, C = video_features.shape

        face_features = face_features.unsqueeze(1).repeat(1, T, 1)
        
        visual_features = torch.cat([face_features, video_features], dim=2)
                
        
        # torch.cat()


def main():
    fp = '/media/ssd/christen-rnd/Experiments/Lip2Speech/vgg_face_recognition/pretrained/vgg_face_torch/VGG_FACE.t7'

    model = NoNameModel(face_reconizer_path=fp)

    video = torch.rand(4, 3, 75, 96, 96)
    face = torch.rand(4, 3, 224, 224)
    speech = torch.rand(1, 16000 * 1)

    outputs = model(video, face, speech)

    print(outputs.shape)


if __name__ == '__main__':
    main()