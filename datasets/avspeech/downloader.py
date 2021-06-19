#!/usr/bin/env python3

import traceback
import os
import sys
import pickle
import youtube_dl
import ffmpeg
from multiprocessing.pool import ThreadPool

try:
    from .preprocess import VidInfo
except:
    from preprocess import VidInfo


ROOT_PATH = '/home/hlcv_team028/Project/Datasets/AVSpeech'


def download(vidinfo):
    yt_base_url = 'https://www.youtube.com/watch?v='

    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        'format': '22/18',
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            meta_info = ydl.extract_info(url=yt_url, download=False)
            download_url = meta_info['url']
    except:
        traceback.print_exc()
        
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg

    vidinfo.out_video_filename = os.path.join(ROOT_PATH, vidinfo.out_video_filename)
    vidinfo.out_audio_filename = os.path.join(ROOT_PATH, vidinfo.out_audio_filename)

    try:
        stream = ffmpeg.input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)

        ffmpeg.output(stream.video, vidinfo.out_video_filename, format='mp4', r=25, vcodec='libx264',
                    crf=18, preset='veryfast', pix_fmt='yuv420p').run_async(overwrite_output=True, quiet=True)
        ffmpeg.output(stream.audio, vidinfo.out_audio_filename, ac=1, acodec='pcm_s16le', ar=16000).run_async(overwrite_output=True, quiet=True)

    except:
        traceback.print_exc()

        return_msg = '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)
        return return_msg

    return '{}, DONE!'.format(vidinfo.yt_id)


def main():
    data_type = sys.argv[1]
    assert data_type in ('test', 'train')

    with open(f'{data_type}.pickle', 'rb') as pickle_file:
        vidinfos = pickle.load(pickle_file)


    results = ThreadPool(8).imap_unordered(download, vidinfos)
    cnt = 0

    for r in results:
        cnt += 1
        print(cnt, '/', len(vidinfos), r)


if __name__ == '__main__':
    main()