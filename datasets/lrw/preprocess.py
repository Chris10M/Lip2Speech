
import os
import sys
import ffmpeg

def preprocess(data_path):
  for root, _, filenames in os.walk(data_path):
    for filename in filenames:
        if filename.endswith('.mp4'):
            print(filename)
            video_path = os.path.join(root, filename)
            try:
                stream = ffmpeg.input(video_path)
                stream  = ffmpeg.output(stream.audio, video_path.replace('mp4','wav'), ac=1, acodec='pcm_s16le', ar=16000)
                ffmpeg.run(stream)

            except Exception as e:
              return e
  return 'DONE!'
def main():
    data_path = sys.argv[1]
    print(preprocess(data_path))


if __name__ == '__main__':
    main()