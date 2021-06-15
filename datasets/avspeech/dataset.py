import random
import torch
import torchvision
import torchaudio

from torch._C import dtype

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from logging import Logger
import imutils
import cv2
import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import face_alignment


class AVSpeech(Dataset):
    def __init__(self, rootpth, face_size=(96, 96), mode='train', demo=False, frame_length=3, *args, **kwargs):
        super(AVSpeech, self).__init__(*args, **kwargs)
        assert mode in ('train', 'test')

        self.rootpth = rootpth

        self.face_recog_resize = transforms.Resize((224, 224))
        self.face_resize = transforms.Resize(face_size)

        self.mode = mode
        self.demo = demo
        
        self.data_path = os.path.join(self.rootpth, mode) 
        
        self.items = dict()
        index = 0
        for root, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                
                if filename.endswith('.mp4'):
                    video_path = os.path.join(root, filename)
                    audio_path = os.path.join(root, filename.replace('.mp4', '.wav'))
                    frame_info_path = os.path.join(root, filename.replace('.mp4', '.json'))
                    
                    if os.path.isfile(audio_path):
                        self.items[index] = [video_path, audio_path, frame_info_path]
                        index += 1

        self.len = index
        self.frame_length = frame_length

    def __len__(self):
        return self.len

    def get_another_item(self):
        return self[random.choice(range(len(self)))]

    def __getitem__(self, idx):
        item = self.items[idx]
        
        video_pth, audio_pth, frame_info_path = item

        speech, sampling_rate = torchaudio.load(audio_pth, num_frames=16000 * self.frame_length)
        assert sampling_rate == 16000
        
        if speech.shape[0] == 0:
            return self.get_another_item()
    
        frames, _, _ = torchvision.io.read_video(video_pth, end_pts=self.frame_length, pts_unit='sec')
        frames = frames[:25 * self.frame_length].permute(0, 3, 1, 2)
        
        with open(frame_info_path, 'r') as json_path:
            frame_info = json.load(json_path)

        N = frames.shape[0]

        faces = list()
        for idx in range(N):
            x1, y1, x2, y2 = frame_info[str(idx)]['face_coords']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            face = frames[idx, :, y1: y2, x1: x2]
            
            faces.append(face)

        if len(faces) == 0:
            return self.get_another_item()

        face_indices = (torch.rand(1) * N).int()
        face_crop = torch.cat([self.face_recog_resize(faces[f_id]).unsqueeze(0) for f_id in face_indices], dim=0)

        lower_faces = torch.cat([self.face_resize(face).unsqueeze(0) for face in faces], dim=0)

        return lower_faces, speech, face_crop


def main():
    cropsize = [384, 384]

    ds = AVSpeech('/media/ssd/christen-rnd/Experiments/Lip2Speech/datasets/avspeech', mode='test')
    dl = DataLoader(ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True)

    from IPython.display import Audio, display

    for frames, speech, faces in dl:
        frames = faces
        print('faces.shape ', faces.shape)
        print('frames[0][0].shape ', frames[0][0].shape)
        print('speech.shape ', speech.shape)

        image = frames[0][0].permute(1, 2, 0).numpy()

        sample_rate = 16000
        effects = [
                    ["lowpass", "-1", "700"], # apply single-pole lowpass filter
                    # ["speed", "0.8"],  # reduce the speed
                                        # This only changes sample rate, so it is necessary to
                                        # add `rate` effect with original sample rate after this.
                    # ["rate", f"{sample_rate}"],
                    # ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]

        aug_speech, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
        speech[0], sample_rate, effects)

        torchaudio.save('test.wav', speech[0], 16000)
        torchaudio.save('aug_speech.wav', aug_speech, 16000)

        # plot_waveform(waveform, sample_rate)
        # plot_specgram(waveform, sample_rate)
        # play_audio(waveform, sample_rate)

        # images = images.numpy()
        # lb = lb.numpy()

        # for image, label in zip(images, lb):
        #     label = ds.vis_label(label)

        cv2.imshow('image', image)

        if ord('q') == cv2.waitKey(0):
            exit()

        exit()

    #     print(torch.unique(label))
    #     print(img.shape, label.shape)


if __name__ == "__main__":
    main()
