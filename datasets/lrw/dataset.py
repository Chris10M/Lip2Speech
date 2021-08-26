import torch
import sys
import torch.nn as nn 
import bz2
import pickle
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


try: 
    from datasets import MelSpectrogram
except: 
    sys.path.extend(['..'])
    from spectograms import MelSpectrogram
    

def loadframes(filename):
    with bz2.BZ2File(filename, 'r') as f:
        data = pickle.load(f)
    
    return np.array([cv2.imdecode(imn, cv2.IMREAD_COLOR)[:, :, ::-1] for imn in data])


def train_collate_fn_pad(batch):
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



class LRW(Dataset):
    def __init__(self, rootpth, face_size=(96, 96), mode='train', demo=False, duration=1, face_augmentation=None, *args, **kwargs):
        super(LRW, self).__init__(*args, **kwargs)
        assert mode in ('train', 'test', 'val')

        self.rootpth = rootpth
        
        self.face_recog_resize = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda im: (im.float() - 127.5) / 128.0),
            ])

        self.face_size = face_size
        self.face_resize = transforms.Compose([
            transforms.Lambda(lambda im: im.float() / 255.0),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        if face_augmentation is None:
            self.face_augmentation = nn.Identity()
        else:
            self.face_augmentation = face_augmentation

        self.melspec_g = MelSpectrogram()

        self.mode = mode
        self.demo = demo
        
        self.items = dict()
        index = 0
        
        with open(os.path.join(rootpth, 'lrw500_detected_face.csv')) as csv_file:
            lines = csv_file.read().splitlines()
        
        for filename in map(lambda l: l.split(',')[0], filter(lambda x: mode == x.split('/')[-2], lines)):
            audio_path = os.path.join(rootpth, 'lipread_audio', f'{filename}.npz')
            face_path =  os.path.join(rootpth, 'LRW_Faces', f'{filename}_face.npz')
            mouth_path =  os.path.join(rootpth, 'LRW_Faces', f'{filename}_mouth.npz')
            
            self.items[index] = [face_path, mouth_path, audio_path]            
            index += 1

        self.len = len(self.items)
        self.duration = duration

        print(f'Size of {type(self).__name__}: {self.len}')
         
    def __len__(self):
        return self.len
    
    def reset_item(self):
        return self[np.random.choice(self.len)]

    def __getitem__(self, idx):
        face_path, mouth_path, audio_path = self.items[idx]

        faces = loadframes(face_path)
        mouth = loadframes(mouth_path)
        audio = np.load(audio_path)['data'][np.newaxis]
            
        faces = torch.from_numpy(faces).permute(0, 3, 1, 2)
        mouth = torch.from_numpy(mouth).permute(0, 3, 1, 2)

        speech = torch.from_numpy(audio)

    
        melspec = self.melspec_g(speech).squeeze(0)
        
        mouth = self.face_resize(mouth)

        face_indices = (torch.rand(2) * len(faces)).int()
        face_crop = torch.cat([self.face_recog_resize(faces[f_id]).unsqueeze(0) for f_id in face_indices], dim=0)

        if self.demo:
            return mouth, speech, melspec, face_crop, (face_path, audio_path)
            
        return mouth, speech, melspec, face_crop


def main(): 
    ds = LRW('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/LRW', mode='test')

    dl = DataLoader(ds,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    collate_fn=train_collate_fn_pad,
                    drop_last=True)

    from IPython.display import Audio, display

    for bdx, batch in enumerate(dl):
        (video, video_lengths), (speeches, audio_lengths), (melspecs, melspec_lengths, mel_gates), faces = batch
        
        frames = video
        print('video.shape', video.shape)
        print('faces.shape ', faces.shape)
        print('frames[0][0].shape ', frames[0][0].shape)
        print('melspecs.shape ', melspecs.shape)
        # print('speech.shape ', speech.shape)

        continue
        B, C, T, H, W = video.shape

        for k in range(B):
            for i in range(T):
                image = frames[k, :, i, :, :].permute(1, 2, 0).numpy()
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                
                print(k, i, image.shape)


                cv2.imshow('image', image[:, :, :: -1])

                if ord('q') == cv2.waitKey(16):
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
