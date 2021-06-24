import torch
import soundfile as sf
import librosa
import os
import shutil


ROOT_PATH = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/GRID'
REQUIRED_SAMPLING_RATE = 16000


def main():
    audio_pths = list()
    for root, _, filenames in os.walk(ROOT_PATH):
        audio_pths.extend([os.path.join(root, filename) for filename in filenames if filename.endswith('wav')])
    
    for idx, audio_pth in enumerate(audio_pths):
        _, SR = sf.read(audio_pth)
        if SR == REQUIRED_SAMPLING_RATE: continue 
        
        speech, sampling_rate = librosa.load(audio_pth, sr=REQUIRED_SAMPLING_RATE)
                
        sf.write('tmp.wav', speech, sampling_rate, subtype='PCM_16')
        shutil.move('tmp.wav', audio_pth)

        print(f'Idx: {idx} of {len(audio_pths)}')


if __name__ == "__main__":
    main()