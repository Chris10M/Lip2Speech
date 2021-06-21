import random
from librosa.filters import mel
import torch
from torch._C import dtype
import torchvision
import torchaudio
import torchvision.transforms as transforms
import cv2
import traceback
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from logging import Logger

from torchvision.transforms.transforms import Lambda

try: from .utils import LinearSpectrogram
except: from utils import LinearSpectrogram


def av_speech_collate_fn_trim(batch):
    "NOTE: WILL NOT WORK WITH THE NEW CODE, HAVE NOT CHANGED IT."
    lower_faces, speeches, melspecs, face_crop = zip(*batch)
    
    N = len(lower_faces)
    max_frames_in_batch = min([l.shape[0] for l in lower_faces])
    max_samples_in_batch = min([s.shape[1] for s in speeches])
    
    trimmed_lower_faces = list()
    trimmed_speeches = list()
    for lower_face, speech, melspec in zip(lower_faces, speeches, melspecs):
        trimmed_lower_faces.append(lower_face[:max_frames_in_batch, :, :, :].unsqueeze(0))
        trimmed_speeches.append(speech[:, :max_samples_in_batch].unsqueeze(0))

    lower_faces_tensor = torch.cat(trimmed_lower_faces, dim=0)
    speeches_tensor = torch.cat(trimmed_speeches, dim=0)

    face_crop_tensor = torch.cat([f.unsqueeze(0) for f in face_crop], dim=0)

    return (lower_faces_tensor, [max_frames_in_batch for _ in range(N)]),\
              (speeches_tensor, [max_samples_in_batch for _ in range(N)]), face_crop_tensor


def av_speech_collate_fn_pad(batch):
    lower_faces, speeches, melspecs, face_crop = zip(*batch)
    
    max_frames_in_batch = max([l.shape[0] for l in lower_faces])
    max_samples_in_batch = max([s.shape[1] for s in speeches])
    max_melspec_samples_in_batch = max([m.shape[1] for m in melspecs])

    padded_lower_faces = torch.zeros(len(lower_faces), max_frames_in_batch, *tuple(lower_faces[0].shape[1:]))
    padded_speeches = torch.zeros(len(speeches), 1, max_samples_in_batch)
    padded_melspecs = torch.zeros(len(melspecs), melspecs[0].shape[0], max_melspec_samples_in_batch)
    mel_gate_padded = torch.zeros(len(melspecs), max_melspec_samples_in_batch)

    video_lengths = list()
    audio_lengths = list()
    melspec_lengths = list()
    for idx, (lower_face, speech, melspec) in enumerate(zip(lower_faces, speeches, melspecs)):
        T = lower_face.shape[0]
        video_lengths.append(T)

        padded_lower_faces[idx, :T, :, :, :] = lower_face

        S = speech.shape[-1]
        audio_lengths.append(S)
        padded_speeches[idx, :, :S] = speech
        
        M = melspec.shape[-1]
        melspec_lengths.append(M)
        padded_melspecs[idx, :, :M] = melspec

        mel_gate_padded[idx, M-1:] = 1.0

    face_crop_tensor = torch.cat([f.unsqueeze(0) for f in face_crop], dim=0)
    padded_lower_faces = padded_lower_faces.permute(0, 2, 1, 3, 4)
    padded_speeches = padded_speeches.squeeze(1) 

    video_lengths = torch.tensor(video_lengths)
    audio_lengths = torch.tensor(audio_lengths)
    melspec_lengths = torch.tensor(melspec_lengths)

    return (padded_lower_faces, video_lengths), (padded_speeches, audio_lengths), (padded_melspecs, melspec_lengths, mel_gate_padded), face_crop_tensor


class AVSpeech(Dataset):
    def __init__(self, rootpth, face_size=(96, 96), mode='train', demo=False, frame_length=3, *args, **kwargs):
        super(AVSpeech, self).__init__(*args, **kwargs)
        assert mode in ('train', 'test')

        self.rootpth = rootpth
        
        self.linear_spectogram = LinearSpectrogram()

        self.face_recog_resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda im: torch.index_select(im.float(), 0, torch.LongTensor([2, 1, 0]))), # RGB2BGR
            transforms.Normalize((129.1863, 104.7624, 93.5940), (1.0, 1.0, 1.0)),
            ])

        self.face_resize = transforms.Compose([
            transforms.Resize(face_size),
            transforms.Lambda(lambda im: im.float() / 255.0),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

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
        
        try:
            speech, sampling_rate = torchaudio.load(audio_pth, num_frames=16000 * self.frame_length)
        except:
            # traceback.print_exc()
            return self.get_another_item()

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
            face_coords = np.array(frame_info[str(idx)]['face_coords'], dtype=np.int)
            face_coords[face_coords < 0] = 0
            x1, y1, x2, y2 = face_coords

            face = frames[idx, :, y1: y2, x1: x2]
            
            if face.shape[1] < 16 or face.shape[2] < 16: return self.get_another_item()

            faces.append(face)

        if len(faces) == 0:
            return self.get_another_item()

        face_indices = (torch.rand(2) * N).int()
        face_crop = torch.cat([self.face_recog_resize(faces[f_id]).unsqueeze(0) for f_id in face_indices], dim=0)
        lower_faces = torch.cat([self.face_resize(face).unsqueeze(0) for face in faces], dim=0)

        melspec = self.linear_spectogram(speech).squeeze(0)

        return lower_faces, speech, melspec, face_crop


def main():    
    ds = AVSpeech('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='test')
    ds[1]

    dl = DataLoader(ds,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=av_speech_collate_fn_pad)

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

        # torchaudio.save('test.wav', speech[0], 16000)
        # torchaudio.save('aug_speech.wav', aug_speech, 16000)

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
