#!/usr/bin/env python3

import random
import re
import torch
import sys
import torch
import os
import json
import numpy as np
import cv2
import pickle
import bz2
import torchaudio
import torchvision.transforms as transforms
import ffmpeg
import torchvision
from multiprocessing.pool import ThreadPool
from fast_detector import FaceDetector
sys.path.extend(['..'])
from spectograms import MelSpectrogram
from face_utils import align_and_crop_face
from facenet_pytorch import MTCNN, InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



ROOT_PATH = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/WILD'
TARGET_FACE = '/home/christen/Downloads/Elon_Musk_Royal_Society.jpg'

WORKERS = 3
SPLIT_SECOND = 2


with torch.no_grad():
    TARGET_FACE_EMBEDDING = InceptionResnetV1(pretrained='vggface2').eval()((transforms.Compose([
    transforms.Lambda(lambda im: (im.float() - 127.5) / 128.0),
    ])(torch.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread(TARGET_FACE), (160, 160)), cv2.COLOR_BGR2RGB))).unsqueeze(0).permute(0, 3, 1, 2))).to(device)


FDS = [FaceDetector(target_face_embedding=TARGET_FACE_EMBEDDING) for _ in range(WORKERS)]
melspectogram = MelSpectrogram().to(device)


def save2pickle(filename, data=None):                                               
    assert data is not None, "data is {}".format(data)                           
    if not os.path.exists(os.path.dirname(filename)):                            
        os.makedirs(os.path.dirname(filename), exist_ok=True)                                   
    
    with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(data, f)


def write_segment(segment_info): 
    idx, video_path, timestamp, save_path = segment_info
    done_path = save_path[:-3] + 'done'
    json_path = save_path[:-3] + 'json'
    audio_path = save_path[:-3] + 'wav'
    mel_path = save_path[:-3] + 'npz'
    face_path = save_path[:-4] + '_face.npz'

    start_time, end_time = timestamp

    if os.path.isfile(done_path):
        print(f'{idx} Already Processed')
        return 

    stream = ffmpeg.input(video_path, ss=start_time, to=end_time)

    ffmpeg.output(stream.audio, audio_path, ac=1, acodec='pcm_s16le', ar=16000).run_async(overwrite_output=True, quiet=True)
    ffmpeg.output(stream, save_path, format='mp4', r=25, vcodec='libx264',
                  crf=18, preset='veryfast', pix_fmt='yuv420p', ac=1, ar=16000).run(overwrite_output=True, quiet=True)

    frames, audio, meta = torchvision.io.read_video(save_path)

    melspec = melspectogram(audio.to(device)).cpu().numpy()
    np.savez_compressed(mel_path, data=melspec)

    face_infos = FDS[idx % WORKERS](frames)

    if face_infos is None: 
        with open(done_path, 'w') as txt_file: ...
        return

    video_info = dict()
    faces = list()
    for idx, face_info in enumerate(face_infos):
        if face_info is None:
            with open(done_path, 'w') as txt_file: ...
            return idx # dont save the video segment.

        box, landmark = face_info
        video_info[idx] = {"face_coords": box.tolist(), "landmarks": landmark.tolist()}

        frame = frames[idx, :, :, :].permute(2, 0, 1)
    
        face = align_and_crop_face(frame, box, landmark).permute(1, 2, 0).numpy()[:, :, ::-1]

        faces.append(face)
    
    faces = [cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1] for im in faces]    
    save2pickle(face_path, data=faces)

    with open(json_path, 'w') as json_file:
        json.dump(video_info, json_file)

    with open(done_path, 'w') as txt_file: ...
    return idx


def split_video(idx, total,video_path, folder_path):
    video_info = ffmpeg.probe(video_path)['format']
    duration = round(float(video_info['duration']), 2)
    
    splits = np.arange(0, duration, SPLIT_SECOND)
    
    # duration - splits[-1]
    segment_infos = list() 
    for i in range(len(splits) - 1):
        timestamp = [splits[i], splits[i + 1]]
        save_path = f'{folder_path}/{splits[i]}.mp4'

        segment_infos.append([i, video_path, timestamp, save_path])

    segment_infos.append([i + 1, video_path, [splits[-1], duration], f'{folder_path}/{splits[-1]}.mp4'])
    
    random.shuffle(segment_infos)

    results = ThreadPool(WORKERS).imap_unordered(write_segment, segment_infos)
    
    cnt = 0
    for r in results:
        print(idx, '/', total, cnt, '/', len(segment_infos))
        cnt += 1
        



def main():
    file_paths = list()
    for root, _, filenames in os.walk(ROOT_PATH):
        file_paths.extend([os.path.join(root, filename) for filename in filenames])
        break
        
    for f_dx, file_path in enumerate(file_paths):
        path = file_path.split('/')[:-1]
        filename = file_path.split('/')[-1]
        
        folder_path = '/'.join(path + [filename.replace('-', '_').replace(' ', '_').replace('.', '_')])
        os.makedirs(folder_path, exist_ok=True)

        print(file_path)
        split_video(f_dx, len(file_paths), file_path, folder_path)


if __name__ == '__main__':
    main()