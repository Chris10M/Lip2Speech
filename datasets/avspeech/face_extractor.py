from operator import pos
import sys
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
from torch._C import dtype
import torchvision
import torch.nn.functional as F
import face_alignment

try:
    from .preprocess import VidInfo
except:
    from preprocess import VidInfo


ROOT_PATH = '/home/hlcv_team028/Project/Datasets/AVSpeech'
BATCH_SIZE = 16


def main():
    folder = sys.argv[1]
    assert folder.endswith(('test', 'train'))

    with open(f'{folder}.pickle', 'rb') as pickle_file:
        vidinfos = pickle.load(pickle_file)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    for vid_idx, vidinfo in enumerate(vidinfos):
        video_pth = os.path.join(ROOT_PATH, vidinfo.out_video_filename)
        
        frame_index = 0

        frames, _, _ = torchvision.io.read_video(video_pth) 
        frames = frames.permute(0, 3, 1, 2)
        N, C, H, W = frames.shape
        
        frame_info = dict()
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
                    
                    distance = np.linalg.norm(np.array([midx, midy]) - np.array(vidinfo.face_point))
                    possible_faces.append([[x1, y1, x2, y2], distance])

                try:
                    closest_face, _ = sorted(possible_faces, key=lambda f: f[-1])[0]
                    frame_info[frame_index] = {'face_coords': closest_face}
                except IndexError:
                    frame_info[frame_index] = {'face_coords': [0, 0, W, H]}
                
                frame_index += 1

        
        json_path = video_pth[:-3] + 'json'
        
        with open(json_path, 'w') as json_file:
            json.dump(frame_info, json_file)


        print(f'{vid_idx} of {len(vidinfos)}', end='\r')


if __name__ == '__main__':
    main()