
import os
import sys
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
from PIL import Image
import numpy as np
import json
import face_alignment
import ffmpeg
import torch.nn.functional as F
from tqdm import tqdm

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def extract_face(video_pth):
  BATCH_SIZE = 8
  frame_index = 0
  frames, _, _ = torchvision.io.read_video(video_pth)
  frames = frames.permute(0, 3, 1, 2)
  N, C, H, W = frames.shape
  frame_info = dict()
  for i in range(0, N, BATCH_SIZE):
      batch_frames = frames[i: i + BATCH_SIZE].float()
      for face in fa.face_detector.detect_from_batch(batch_frames):
          x1, y1, x2, y2, score = face[0]
          frame_info[frame_index] = {'face_coords': [x1, y1, x2, y2]}
          frame_index += 1
  json_path = video_pth[:-3] + 'json'
  with open(json_path, 'w') as json_file:
      json.dump(frame_info, json_file)

def preprocess(data_path, split):
  for word in tqdm(os.listdir(data_path)):
    filenames = os.listdir(os.path.join(data_path, word, split))
    videos = [f for f in filenames if f.endswith('.mp4')]
    for filename in (videos):
        video_path = os.path.join(data_path, word, split, filename)
        try:
            # Separate Audio from video 
            stream = ffmpeg.input(video_path)
            stream  = ffmpeg.output(stream.audio, video_path.replace('mp4','wav'), ac=1, acodec='pcm_s16le', ar=16000)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            # Extract Face
            extract_face(video_path)
        except Exception as e:
          return e
        
  return 'DONE!'
def main():
    data_path = sys.argv[1]
    split = sys.argv[2]

    print(preprocess(data_path, split))


if __name__ == '__main__':
    main()
