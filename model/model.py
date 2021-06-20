import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import AudioExtractor, VideoExtractor, VGG_16, Decoder
except ModuleNotFoundError: 
    from modules import AudioExtractor, VideoExtractor, VGG_16, Decoder


class NoNameModel(nn.Module):
    def __init__(self, face_reconizer_path, video_input_size=96, train=False, wav2vec_path=''):
        super().__init__()

        self.vgg_face = VGG_16(face_reconizer_path)
        self.vgg_face.freeze_backbone()

        self.video_fe = VideoExtractor(video_input_size)

        self.decoder = Decoder()
        
        self._train_mode = train

        if train:   
            assert os.path.isfile(wav2vec_path)

            self.wav2vec = AudioExtractor(wav2vec_path)

        self.video_pool = torch.nn.AvgPool2d((2, 1), stride=(2, 1))

    def forward(self, video_frames, face_frames, audio_frames, melspecs, video_lengths, audio_lengths, melspec_lengths):
        video_features = self.video_fe(video_frames)
        
        video_features = self.video_pool(video_features) # Assuming each lip-movement takes at least 2 frame. Intution: reduce frames to help LSTM to capture properly(longer dependacy issue) and also improve compute performance. 
        video_lengths = video_lengths // 2

        face_pair_1 = self.vgg_face(face_frames[:, 0, :, :, :])
        # face_pair_2 = self.vgg_face(face_frames[:, 1, :, :, :])

        face_features = face_pair_1

        audio_identity_features = self.wav2vec.identity_features(audio_frames)
        
        N, T, C = video_features.shape

        face_features = face_features.unsqueeze(1).repeat(1, T, 1)
        
        visual_features = torch.cat([face_features, video_features], dim=2)

        outputs = self.decoder(visual_features, melspecs, video_lengths, melspec_lengths)

        return outputs, (face_pair_1, audio_identity_features)


def get_network(mode):
    fp = '/media/ssd/christen-rnd/Experiments/Lip2Speech/vgg_face_recognition/pretrained/vgg_face_torch/VGG_FACE.t7'
    wav2vec_path='/media/ssd/christen-rnd/Experiments/Lip2Speech/wav2vec_large.pt'

    assert mode in ('train', 'test')

    return NoNameModel(face_reconizer_path=fp, wav2vec_path=wav2vec_path, train=(mode == 'train'))


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
