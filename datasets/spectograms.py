import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torchaudio
import torchaudio.transforms as T 
import sys; 
try:
    sys.path.extend(['../..', '..'])
    from hparams import create_hparams
except:
    from hparams import create_hparams



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

def spectral_de_normalize(magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output
    

class MelSpectrogram(torch.nn.Module):
    def __init__(self, hparams=create_hparams()):        
        super(MelSpectrogram, self).__init__()

        self.mel_spec = T.MelSpectrogram(
                                        sample_rate=hparams.sampling_rate, 
                                        n_fft=hparams.filter_length, 
                                        win_length=hparams.win_length, 
                                        hop_length=hparams.hop_length,
                                        f_min=hparams.mel_fmin,
                                        f_max=hparams.mel_fmax,
                                        n_mels=hparams.n_mel_channels
                                    )


    def forward(self, waveform):
        melspec = self.mel_spec(waveform)
        melspec = spectral_normalize(melspec)

        return melspec


class Spec2Audio(torch.nn.Module):
    def __init__(self, hparams=create_hparams()):
        super(Spec2Audio, self).__init__()

        self.griffin_lim = T.GriffinLim(n_fft=hparams.filter_length,
                                        win_length=hparams.win_length,
                                        hop_length=hparams.hop_length)

    def forward(self, spec):
        return self.griffin_lim(spec)


class MelSpec2Audio(torch.nn.Module):
    def __init__(self, hparams=create_hparams(), max_iters=256, ):
        super(MelSpec2Audio, self).__init__()
        
        self.inv_mel_spec = T.InverseMelScale(sample_rate=hparams.sampling_rate,
                                              n_mels=hparams.n_mel_channels, 
                                              f_min=hparams.mel_fmin,
                                              f_max=hparams.mel_fmax,
                                              n_stft=hparams.filter_length // 2 + 1,
                                              max_iter=max_iters)

        self.griffin_lim = T.GriffinLim(n_fft=hparams.filter_length,
                                        win_length=hparams.win_length,
                                        hop_length=hparams.hop_length,
                                        n_iter=max_iters)

    def forward(self, melspec):
        melspec = spectral_de_normalize(melspec)

        return self.griffin_lim(self.inv_mel_spec(melspec))



def main():
    import sys; sys.path.append('../../model/modules/tacotron2')
    from hparams import create_hparams
    # from layers import TacotronSTFT
    # from stft import STFT
    


    hparams = create_hparams()

    file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech/test/-6d8rnJDiO0_120.220000_125.025000.wav'
    
    audio, sampling_rate = torchaudio.load(file)
    assert hparams.sampling_rate == sampling_rate

    m = T.MelSpectrogram(sample_rate=hparams.sampling_rate, 
                     n_fft=hparams.filter_length, 
                     win_length=hparams.win_length, 
                     hop_length=hparams.hop_length,
                     f_min=hparams.mel_fmin,
                     f_max=hparams.mel_fmax,
                     n_mels=hparams.n_mel_channels)
    melspec = m(audio)
    
    # n_stft: int,
    #              n_mels: int = 128,
    #              sample_rate: int = 16000,
    #              f_min: float = 0.,
    #              f_max: Optional[float] = None,
    #              max_iter: int = 100000,
    #              tolerance_loss: float = 1e-5,
    #              tolerance_change: float = 1e-8,
    #              sgdargs: Optional[dict] = None,
    #              norm: Optional[str] = None,
    #              mel_scale: str = "htk"
    inverse_mel_pred = T.InverseMelScale(sample_rate=hparams.sampling_rate,
                                         n_mels=hparams.n_mel_channels, 
                                         f_min=hparams.mel_fmin,
                                         f_max=hparams.mel_fmax,
                                         n_stft=hparams.filter_length // 2 + 1,
                                         max_iter=256)(melspec)
    print(inverse_mel_pred.shape)

    print(melspec.shape)
    torchaudio.save('test.wav', audio, hparams.sampling_rate)

    spec =  MelSpectrogram(hparams)(audio)
    reconstructed_audio = MelSpec2Audio(hparams)(spec)
    torchaudio.save('test_ms_recon.wav', torch.tensor(reconstructed_audio), hparams.sampling_rate)

    # reconstructed_audio = Spec2Audio(hparams)(inverse_mel_pred)
    # torchaudio.save('test_oms_recon.wav', torch.tensor(reconstructed_audio), hparams.sampling_rate)


if __name__ == "__main__":
    main()
