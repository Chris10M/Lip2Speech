import random
import torch
import glob
import sys
import torch.nn as nn 
import torchvision
import ffmpeg
import torchaudio
import torchvision.transforms as transforms
import cv2
import math
import collections
import traceback
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from logging import Logger

from torchvision.transforms.transforms import Lambda
from face_utils import align_and_crop_face 
# from fast_detector import TorchFaceDetector as FaceDetector
# from fast_detector import ONNXFaceDetector as FaceDetector


def av_speech_face_collate_fn(batch):
    speeches, faces = zip(*batch)
    
    min_samples_in_batch = min([s.shape[1] for s in speeches])
    
    trimmed_speeches = torch.zeros(len(speeches), min_samples_in_batch)
    
    for idx, speech in enumerate(speeches):
        S = min_samples_in_batch
        
        trimmed_speeches[idx, :S] = speech[:, :S]
        
    faces_tensor = torch.cat([f.unsqueeze(0) for f in faces], dim=0)

    return trimmed_speeches, faces_tensor


def x_round(x):
    return x * 4 / 4


class AVSpeechFace(Dataset):
    def __init__(self, rootpth, mode='train', demo=False, duration=2, face_augmentation=None, *args, **kwargs):
        super(AVSpeechFace, self).__init__(*args, **kwargs)
        assert mode in ('train', 'test')

        self.rootpth = os.path.join(rootpth, mode)
        
        self.face_recog_resize = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda im: (im.float() - 127.5) / 128.0),
            ])


        self.face_augmentation = transforms.RandomHorizontalFlip()
        # self.face_detector = FaceDetector()

        self.mode = mode
        self.demo = demo
        
        self.items = dict()
        index = 0

        all_wavs_in_root = set(glob.glob(f'{self.rootpth}/*.wav'))
        all_jsons_in_root = set(glob.glob(f'{self.rootpth}/*.json'))
        for video_paths in [glob.glob(f'{self.rootpth}/*.{fmt}') for fmt in ('mp4', 'mov', 'mpg')]:
            for video_path in video_paths:    
                audio_path = video_path[:-3] + 'wav'
                json_path = video_path[:-3] + 'json'
                    
                if json_path in all_jsons_in_root and audio_path in all_wavs_in_root:
                    self.items[index] = [video_path, audio_path, json_path]
                        
                    index += 1
            

        self.len = len(self.items)
        self.duration = duration

        print(f'Size of {type(self).__name__}: {self.len}')

        random.shuffle(self.items)

        self.random_indices = np.random.choice(len(self), 2 * len(self)).tolist()
        self.index = -1

        self.frame_count_per_video = dict()
        self.invalid_frame = collections.defaultdict(set)
        self.invalid_audio = set()

    def __len__(self):
        return self.len
    
    def reset_item(self):
        if self.index < 0:
            random.shuffle(self.random_indices)
            self.index = len(self.random_indices) - 1

        idx = self.random_indices[self.index]; self.index -= 1 
        
        return self[idx]
        
    def __getitem__(self, idx):
        video_path, audio_path, json_path = self.items[idx]

        if audio_path in self.invalid_audio: return self.reset_item()

        frames_path = video_path[:-4] + '/frames'

        if frames_path not in self.frame_count_per_video:
            total_frames = len(glob.glob(f'{frames_path}/*.jpg'))
            self.frame_count_per_video[frames_path] = total_frames

        total_frames = self.frame_count_per_video[frames_path]

        if total_frames == 0: return self.reset_item()

        fps = 25
        end_time =  total_frames / fps
        duration = self.duration
        
        try:
            # end_time = x_round(float(ffmpeg.probe(video_path)['format']['duration']))
            start_time = random.choice(np.arange(0, end_time, 0.25))
        except: 
            return self.reset_item()

        if (start_time + duration) > end_time:
            start_time = start_time - duration

        start_time = x_round(max(0, start_time))
        end_time = min(end_time, start_time + duration)
        duration = x_round(end_time - start_time)

        frame_time = start_time + np.random.uniform(0, 0.25, 1)
        absolute_frame_idx = str(int(frame_time * 25))

        if absolute_frame_idx in self.invalid_frame[json_path]: return self.reset_item()
        
        with open(json_path, 'r') as json_file:
            frame_info = json.load(json_file)
                
        if absolute_frame_idx not in frame_info: 
            self.invalid_frame[json_path].add(absolute_frame_idx)

            return self.reset_item()
        
        face_coords = frame_info[absolute_frame_idx]['face_coords']

        x1, y1, x2, y2 = face_coords
        H, W, = y2 - y1, x2 - x1
                
        if H > 75 and W > 75 and (H + W) > 150: pass
        else: 
            self.invalid_frame[json_path].add(absolute_frame_idx)
            return self.reset_item()


        frame = cv2.imread(f'{frames_path}/{absolute_frame_idx.zfill(5)}.jpg')
        if frame is None:
            self.invalid_frame[json_path].add(absolute_frame_idx) 
            return self.reset_item()

        landmarks = frame_info[absolute_frame_idx]['landmarks']

        frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = align_and_crop_face(frame, face_coords, landmarks)

        if face is None:
            self.invalid_frame[json_path].add(absolute_frame_idx) 
            return self.reset_item()
        
        face = self.face_recog_resize(face)
        face = self.face_augmentation(face)

        try:
            speech, sampling_rate = torchaudio.load(audio_path, frame_offset=int(16000 * start_time), 
                                                               num_frames=int(16000 * duration), normalize=True, format='wav')                       
        except:
            self.invalid_audio.add(audio_path)
            return self.reset_item()
        
        assert sampling_rate == 16000
        
        if speech.shape[1] == 0:
            self.invalid_audio.add(audio_path)
            return self.reset_item()


        return speech, face


def main():    
    ds = AVSpeechFace('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='train', duration=1)

    dl = DataLoader(ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=av_speech_face_collate_fn)

    from IPython.display import Audio, display

    for bdx, batch in enumerate(dl):
        trimmed_speeches, faces = batch
        
        print('trimmed_speeches.shape', trimmed_speeches.shape)
        print('faces.shape ', faces.shape)


        B = faces.shape[0]

        for k in range(B):
            face = faces[k, :, :, :].permute(1, 2, 0).numpy()
            face = ((face * 128.0) + 127.5).astype(dtype=np.uint8)

            cv2.imshow('face', face[:, :, :: -1])

            if ord('q') == cv2.waitKey(0):
                exit()

        # sample_rate = 16000
        # effects = [
        #             ["lowpass", "-1", "700"], # apply single-pole lowpass filter
        #             # ["speed", "0.8"],  # reduce the speed
        #                                 # This only changes sample rate, so it is necessary to
        #                                 # add `rate` effect with original sample rate after this.
        #             # ["rate", f"{sample_rate}"],
        #             # ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        # ]

        # aug_speech, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
        # speech[0], sample_rate, effects)

        # torchaudio.save('test.wav', speech[0], 16000)
        # torchaudio.save('aug_speech.wav', aug_speech, 16000)

        # plot_waveform(waveform, sample_rate)
        # plot_specgram(waveform, sample_rate)
        # play_audio(waveform, sample_rate)

        # images = images.numpy()
        # lb = lb.numpy()

        # for image, label in zip(images, lb):
        #     label = ds.vis_label(label)


    #     print(torch.unique(label))
    #     print(img.shape, label.shape)


if __name__ == "__main__":
    main()
