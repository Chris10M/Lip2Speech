import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T 


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


def get_audio_from_mel(hparams, melspec):
    # hparams.filter_length, hparams.hop_length, hparams.win_length,
    # hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    # hparams.mel_fmax, hparams.max_wav_value

    import librosa
    # from librosa.filters import mel

    audio = librosa.feature.inverse.mel_to_audio(melspec.numpy(), 
                                         sr=hparams.sampling_rate,
                                         n_fft=hparams.filter_length, 
                                         hop_length=hparams.hop_length, 
                                         win_length=hparams.win_length, 
                                         fmin=hparams.mel_fmin, fmax=hparams.mel_fmax,
                                         window='hann', 
                                         center=True, 
                                         pad_mode='reflect', 
                                         power=2.0, 
                                         n_iter=32, 
                                         length=None)
    audio *= hparams.max_wav_value

    
    # print(audio.shape)
    # iMelScale = T.InverseMelScale(n_stft=hparams.filter_length, n_mels=hparams.n_mel_channels, sample_rate=hparams.sampling_rate, 
    #                              f_min=hparams.mel_fmin, f_max=hparams.mel_fmax)

    # sft = iMelScale(melspec)
    # print(sft.shape)
    # # max_iter: int = 100000, tolerance_loss: float = 1e-05, tolerance_change: float = 1e-08, sgdargs: Optional[dict] = None, norm: Optional[str] = None, mel_scale: str = 'htk'

    # griffin_lim = T.GriffinLim(n_fft=hparams.filter_length, win_length=hparams.win_length, hop_length=hparams.hop_length)
    # waveform = griffin_lim(sft)

    return audio


def main():
    import sys; sys.path.append('../../model/modules/tacotron2')
    from hparams import create_hparams
    from layers import TacotronSTFT

    hparams = create_hparams()
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                        hparams.mel_fmax, hparams.max_wav_value)

    file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech/test/_0kBqBMNfOc_90.000000_97.520000.wav'
    
    audio, sampling_rate = torchaudio.load(file)
    assert hparams.sampling_rate == sampling_rate

    torchaudio.save('test.wav', audio, hparams.sampling_rate)

    melspec = get_mel_from_file(stft, file)
    print(melspec.shape)

    # melspec = stft.spectral_de_normalize(melspec)
    reconstructed_audio = get_audio_from_mel(hparams, melspec)
    torchaudio.save('test_recon.wav', torch.tensor(reconstructed_audio), hparams.sampling_rate)


if __name__ == "__main__":
    main()
