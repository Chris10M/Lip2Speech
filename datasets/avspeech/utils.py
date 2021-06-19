from librosa.filters import mel
import numpy as np
import torch
import torchaudio


def get_mel(stft, audio):
    audio = audio.squeeze(0)

    audio_norm = audio / stft.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    return melspec


def get_mel_from_file(stft, filename):    
    audio, sampling_rate = torchaudio.load(filename)
    audio = audio.squeeze(0)

    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / stft.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    return melspec


def main():
    import sys; sys.path.append('../../model/modules/tacotron2')
    from hparams import create_hparams
    from layers import TacotronSTFT

    hparams = create_hparams()
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                        hparams.mel_fmax, hparams.max_wav_value)

    file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech/test/_0kBqBMNfOc_90.000000_97.520000.wav'
    melspec = get_mel_from_file(stft, file)
    print(melspec.shape)


if __name__ == "__main__":
    main()
