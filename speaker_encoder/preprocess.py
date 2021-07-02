#!/usr/bin/env python3
from glob import glob
from matplotlib.pyplot import imshow
import torch
import sys
import torch
import os
import json
import numpy as np
import cv2
import random
import torchaudio
import youtube_dl
import ffmpeg
import torchvision
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor
from fast_detector import ONNXFaceDetector as FaceDetector


ROOT_PATH = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech/train'
WORKERS = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FDS = [FaceDetector() for _ in range(WORKERS)]


def video_to_frames(segment_info, ret_frames=False):
    idx, video_path, audio_path, json_path = segment_info

    frames_path = video_path[:-4] + '/frames'
    os.makedirs(frames_path, exist_ok=True)

    frames, audio, meta = torchvision.io.read_video(video_path)

    if len(glob(f'{frames_path}/*.jpg')) == frames.shape[0]:
        print(f'{idx} Already Processed')

        if ret_frames: return frames
        return idx

    for i in range(frames.shape[0]):
        frame = frames[i].numpy()[:, :, ::-1]
        
        frame_path = f'{frames_path}/{str(i).zfill(5)}.jpg'

        cv2.imwrite(frame_path, frame,  [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
    if ret_frames: return frames

    return idx


def write_segment(segment_info): 
    idx, video_path, audio_path, json_path = segment_info
    
    frames = video_to_frames(segment_info, ret_frames=True)

    if os.path.isfile(json_path):
        print(f'{idx} Already Processed')
        return 

    # frames, audio, meta = torchvision.io.read_video(video_path)

    face_infos = FDS[idx % WORKERS](frames)

    video_info = dict()
    for idx, face_info in enumerate(face_infos):
        if face_info is None:
            return idx # dont save the video segment.

        box, landmark = face_info
        video_info[idx] = {"face_coords": box.tolist(), "landmarks": landmark.tolist()}

    with open(json_path, 'w') as json_file:
        json.dump(video_info, json_file)

    return idx
    

def main():
    count = 0
    file_paths = list()

    all_audio_paths = set(glob(f'{ROOT_PATH}/*.wav'))
    all_json_paths = set(glob(f'{ROOT_PATH}/*.json'))
    for video_paths in [glob(f'{ROOT_PATH}/*.{fmt}') for fmt in ('mp4', 'mov', 'mpg')]:
        for video_path in video_paths:
            json_path = video_path[:-3] + 'json'
            audio_path = video_path[:-3] + 'wav'

            if audio_path in all_audio_paths:
                file_paths.append([count, video_path, audio_path, json_path])
                count += 1

    
    random.shuffle(file_paths)

    # cnt = 0
    # for i in ThreadPool(WORKERS).imap_unordered(video_to_frames, file_paths):
    #     print(cnt, '/', count)
    #     cnt += 1

    
    cnt = 0
    for r in ThreadPool(WORKERS).imap_unordered(write_segment, file_paths):
        print(cnt, '/', count)
        cnt += 1
    

if __name__ == '__main__':
    main()