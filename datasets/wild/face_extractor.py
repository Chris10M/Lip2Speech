#!/usr/bin/env python3

from operator import pos
from posixpath import join
import sys
import time
from logging import Logger
import imutils
import cv2
import pickle
import os.path as osp
import os
from PIL import Image
import numpy as np
import json
from numpy.core.fromnumeric import sort
import torch
from torch._C import dtype
import torchvision
import torch.nn.functional as F
import face_alignment


ROOT_PATH = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/GRID'
BATCH_SIZE = 16


def get_face(fa, frames, N, H, W, frame_info):    
    frame_index = 0
    for i in range(0, N, BATCH_SIZE):
        batch_frames = frames[i: i + BATCH_SIZE].float()

        resized_frames =  F.interpolate(batch_frames, (640, 640), mode='bilinear')
        _, _, rH, rW = resized_frames.shape
                    
        for faces in fa.face_detector.detect_from_batch(resized_frames):
            
            possible_faces = list()
            for face in faces:
                x1, y1, x2, y2, score = face

                x1, x2 = x1 * (W / rW), x2 * (W / rW)
                y1, y2 = y1 * (H / rH), y2 * (H / rH)

                fH = y2 - y1
                fW = x2 - x1

                if fH < 16 or fW < 16: continue

                midx = (x1 + (x2 - x1) / 2.0) / W 
                midy = (y1 + (y2 - y1) / 2.0) / H
                
                distance = np.linalg.norm(np.array([midx, midy]) - np.array([0.5, 0.5]))
                possible_faces.append([[x1, y1, x2, y2], distance])

            try:
                closest_face, _ = sorted(possible_faces, key=lambda f: f[-1])[0]
                frame_info[frame_index] = {'face_coords': closest_face}
            except IndexError:
                frame_info[frame_index] = {'face_coords': [0, 0, W, H]}
            
            frame_index += 1
    
    return frame_info


def get_landmarks(fa, frames, N, H, W, frame_info):
    for f_id, frame in frame_info.items():
        face = np.array([frame['face_coords']])
        face[face < 0] = 0
        frame_info[f_id]['face_coords'] = face[0].astype(dtype=np.int32).tolist()


        frame = frames[int(f_id)].permute(1, 2, 0)

        landmarks = fa.get_landmarks_from_image(frame, face)[0]
        frame_info[f_id]['landmarks'] = landmarks.astype(dtype=np.int32).tolist()

    return frame_info        


def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    video_pths = list()
    for root, _, filenames in os.walk(ROOT_PATH):
        video_pths.extend([os.path.join(root, filename) for filename in filenames if filename.endswith(('mov', 'mpg'))])
        
    for vid, video_pth in enumerate(video_pths):
        json_path = video_pth[:-3] + 'json'
        
        frame_info = dict()
        if os.path.isfile(json_path):
            with open(json_path, 'r') as json_file:
                frame_info = json.load(json_file)
            if frame_info and len(frame_info) and 'face_coords' in frame_info['0'] and 'landmarks' in frame_info['0']: continue
        
        frames, _, _ = torchvision.io.read_video(video_pth) 
        frames = frames.permute(0, 3, 1, 2)
        N, C, H, W = frames.shape
        
        if frame_info and len(frame_info) and 'face_coords' in frame_info['0']:
            ...
        else:
            frame_info = dict()
            frame_info = get_face(fa, frames, N, H, W, frame_info)

        frame_info = get_landmarks(fa, frames, N, H, W, frame_info)
        
        with open(json_path, 'w') as json_file:
            json.dump(frame_info, json_file)


        print(f'{vid} of {len(video_pths)}', end='\r')


if __name__ == '__main__':
    main()