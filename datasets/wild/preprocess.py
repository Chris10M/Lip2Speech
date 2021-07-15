#!/usr/bin/env python3

import random
import time
import torch
import sys
import torch
import os
import json
import numpy as np
import cv2
import pickle
import bz2
from torch.serialization import save
import torchaudio
import hashlib
import torchvision.transforms as transforms
import ffmpeg
import torchvision
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fast_detector import FaceDetector
sys.path.extend(['..'])
from spectograms import MelSpectrogram
from face_utils import align_and_crop_face
from facenet_pytorch import MTCNN, InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



ROOT_PATH = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/WILD'
TARGET_FACE = '/home/christen/Downloads/Elon_Musk_Royal_Society.jpg'

WORKERS = 1
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



def write_video(video_path, audio_path, save_path):
    stream = ffmpeg.input(video_path)

    ffmpeg.output(stream.audio, audio_path, ac=1, acodec='pcm_s16le', ar=16000).run_async(overwrite_output=True, quiet=True)
    ffmpeg.output(stream, save_path, format='mp4', r=25, vcodec='libx264',
                  crf=18, preset='veryfast', pix_fmt='yuv420p', ac=1, ar=16000).run(overwrite_output=True, quiet=True)


def write_segment(segment_info): 
    idx, video_path, save_path = segment_info
    done_path = save_path[:-3] + 'done'
    json_path = save_path[:-3] + 'json'
    audio_path = save_path[:-3] + 'wav'
    mel_path = save_path[:-3] + 'npz'
    face_path = save_path[:-4] + '_face.npz'

    if os.path.isfile(done_path):
        print(f'{idx} Already Processed')
        return 

    write_video(video_path, audio_path, save_path)
    
    frames, audio, meta = torchvision.io.read_video(save_path)

    try:
        melspec = melspectogram(audio.to(device)).cpu().numpy()
    except:
        with open(done_path, 'w') as txt_file: ...
        return

    np.savez_compressed(mel_path, data=melspec)

    try:
        face_infos = FDS[idx % WORKERS](frames)
    except:
        with open(done_path, 'w') as txt_file: ...
        return

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


def split_video(idx, total, video_path, folder_path):
    video_info = ffmpeg.probe(video_path)['format']
    duration = round(float(video_info['duration']), 2)
    
    splits = np.arange(0, duration, SPLIT_SECOND)
    
    segment_infos = list()
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    index = 0
    while True:
        segment_path = f'temp/{video_hash}{str(index).zfill(6)}.mp4'

        if not os.path.isfile(segment_path): break

        save_path = f'{folder_path}/{splits[index]}.mp4'
        
        segment_infos.append([index, segment_path, save_path])

        index += 1


    random.shuffle(segment_infos)

    total_segments = len(segment_infos)

    # for cnt, segment_info in enumerate(segment_infos): 
    #     print(idx, '/', total, cnt, '/', total_segments)
    #     write_segment(segment_info)

    for cnt, _ in enumerate(ThreadPoolExecutor(4).map(write_segment, segment_infos)):
        print(idx, '/', total, cnt, '/', total_segments)
        



def main():
    os.makedirs('temp', exist_ok=True)

    file_paths = list()
    for root, _, filenames in os.walk(ROOT_PATH):
        file_paths.extend([os.path.join(root, filename) for filename in filenames])
        break
    
    random.shuffle(file_paths)

    for f_dx, file_path in enumerate(file_paths):
        path = file_path.split('/')[:-1]
        filename = file_path.split('/')[-1]
        
        folder_path = '/'.join(path + [filename.replace('-', '_').replace(' ', '_').replace('.', '_')])
        os.makedirs(folder_path, exist_ok=True)

        save_prefix = hashlib.md5(file_path.encode()).hexdigest()
        os.system(f'ffmpeg -i "{file_path}" -c copy -map 0 -segment_time {SPLIT_SECOND} -f segment -reset_timestamps 1 temp/{save_prefix}%06d.mp4')

        print(file_path)
        split_video(f_dx, len(file_paths), file_path, folder_path)


if __name__ == '__main__':
    main()