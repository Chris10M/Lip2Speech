import random
import torch
import torchvision
import torchaudio

from torch._C import dtype

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
import json
import torch.nn.functional as F



class LRW(Dataset):

    def __init__(self, rootpth, face_size=(96, 96), mode='train', demo=False, frame_length=3, *args, **kwargs):
        super(LRW, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')

        self.rootpth = rootpth

        self.face_recog_resize = transforms.Resize((224, 224))
        self.face_resize = transforms.Resize(face_size)

        self.mode = mode
        self.demo = demo
        
        self.data_path = os.path.join(self.rootpth) 
        
        self.items = dict()
        index = 0
        for root, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.endswith('.mp4') and mode in root:
                    video_path = os.path.join(root, filename)
                    audio_path = os.path.join(root, filename.replace('.mp4', '.wav'))
                    frame_info_path = os.path.join(root, filename.replace('.mp4', '.json'))
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
  
    
        frames, _, _ = torchvision.io.read_video(video_pth)
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
        
        return lower_faces, speech, face_crop