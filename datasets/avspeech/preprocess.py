'''
        YouTube ID, start segment, end segment, X coordinate, Y coordinate

'''
import sys
import os
from multiprocessing.pool import ThreadPool
import re
import pickle

from youtube_meta_scrapper import meta_scrapper

import youtube_dl
import ffmpeg


class VidInfo:
    def __init__(self, yt_id, start_time, end_time, face_x, face_y, outdir):
        self.yt_id = yt_id
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.out_video_filename = os.path.join(outdir, yt_id + '_' + start_time + '_' + end_time + '.mp4')
        self.out_audio_filename = os.path.join(outdir, yt_id + '_' + start_time + '_' + end_time + '.wav')

        self.face_point = (float(face_x), float(face_y))


def main():
    split = sys.argv[1]
    csv_file = 'avspeech_{}.csv'.format(split)
    
    out_dir = split

    os.makedirs(out_dir, exist_ok=True)

    with open(csv_file, 'r') as f:
        lines = f.readlines()
        lines = [x.split(',') for x in lines]
        vidinfos = {x[0]: VidInfo(x[0], x[1], x[2], x[3], x[4], out_dir) for x in lines}

    english_vidinfos = list()
    
    keys = list(vidinfos.keys())
    for i in range(0, len(keys), 50):
        batch = keys[i: i + 50]

        metas = meta_scrapper(batch)
        
        for meta in metas['items']:
            y_id = meta['id']
                
            video_meta = meta['snippet']

            english = False
            if 'defaultAudioLanguage' in video_meta:
                if video_meta['defaultAudioLanguage'] == 'en':
                    english = True
            
            if 'defaultLanguage' in video_meta:
                if video_meta['defaultLanguage'] == 'en':
                    english = True

            if english:
                english_vidinfos.append(vidinfos[y_id])

        with open(f'{split}.pickle', 'wb') as pickle_file:
            pickle.dump(english_vidinfos, pickle_file)

        print(f'Done {i} of {len(keys)}')


if __name__ == '__main__':
    main()