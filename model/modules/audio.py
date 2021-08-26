import torch
import torchaudio.transforms as AT
import torch.nn.functional as F
from torch import nn
import numpy as np
import fairseq


try:
    from hparams import create_hparams
except:
    import sys; sys.path.append('../..')
    from hparams import create_hparams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudioExtractor(nn.Module):
    def __init__(self, path, fine_tuning=True):
        super().__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        model = model[0]

        if fine_tuning: 
            # model.eval()
            
            for param in model.parameters():
                param.requires_grad = False
                
        self.model = model
        
        self.identity_projection = nn.Sequential(
            nn.Dropout(0.2),
            
            nn.Linear(512, 512),
            nn.Linear(512, 512),
        )

    def features(self, x):
        z = self.model.feature_extractor(x)
        c = self.model.feature_aggregator(z)
        
        return c
        
    def identity_features(self, x):
        N = x.shape[0]

        x = self.features(x)

        x = F.adaptive_avg_pool1d(x, 1).view(N, -1)
        x = self.identity_projection(x)

        return x


class SpecEncoder(nn.Module):
    def make_conv(self, in_chns, out_chns, kernel_size, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_chns, out_chns, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_chns),
            nn.ReLU()
        )
        
    def __init__(self):
        super().__init__()

        hparams  = create_hparams()
        in_chns = hparams.filter_length // 2 + 1

        #N x 2 x 598 × 257
        #N x C x T x F

        self.fe = nn.Sequential(
            self.make_conv(in_chns, 64, 4),
            self.make_conv(64, 64, 4),
            self.make_conv(64, 128, 4),
            nn.MaxPool1d(2, 2),

            self.make_conv(128, 128, 4),
            nn.MaxPool1d(2, 2),

            self.make_conv(128, 256, 4),
            nn.MaxPool1d(2, 2),
            
            self.make_conv(256, 512, 4),
            nn.MaxPool1d(2, 2),

            self.make_conv(512, 512, 4),
            self.make_conv(512, 512, 4, 2),
            self.make_conv(512, 512, 4, 2),

            nn.AdaptiveAvgPool1d(1),
        )   

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            ReLU(),
            nn.Linear(512, 512),            
        )

    def forward(self, x):
        x = self.fe(x)
        N, C, T = x.shape

        return self.fc(x.view(N, C))


class SpeakerEncoder(nn.Module):
    def __init__(self, state_dict=None):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=40,
                            hidden_size=256, 
                            num_layers=3, 
                            batch_first=True)

        self.linear = nn.Linear(in_features=256, out_features=256)
                
        for name, p in self.named_parameters():
            p.requires_grad_(False)

        self.mel_spec = AT.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40)

        if state_dict is None:
            state_dict = torch.load('speaker_encoder.pt', map_location=device)['model_state']        

        self.load_state_dict(state_dict, strict=True)


    def forward(self, utterances, hidden_init=None):
        utterances = self.mel_spec(utterances).permute(0, 2, 1)
        
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # embeds_raw = F.relu(self.linear(hidden[-1]))
        embeds_raw = (self.linear(hidden[-1]))

        # embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds_raw
        
    def inference(self, x):
        if self.training:
            self = self.eval()
        
        with torch.no_grad():
            embeds_raw = F.relu(self(x))
            return F.normalize(embeds_raw, p=2, dim=1)


def main():
    import torchaudio

    mel_window_length = 25  # In milliseconds
    mel_window_step = 10    # In milliseconds
    mel_n_channels = 40
    ## Audio
    sampling_rate = 16000
    
    n_fft = int(sampling_rate * mel_window_length / 1000)
    hop_length = int(sampling_rate * mel_window_step / 1000)
    n_mels = mel_n_channels
    print(n_fft, hop_length, n_mels)
    # torch.Size([1, 513, 1000])

    # model = AudioExtractor('/media/ssd/christen-rnd/Experiments/Lip2Speech/wav2vec_large.pt')
    # speech = torch.rand(1, 16000 * 1)
    # speech = torch.cat([speech, speech], 0)

    # outputs = model.identity_features(speech)
    # print(outputs.shape)
    
    model = SpeakerEncoder()

    file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/DL/Deep_Learning(CS7015)___Lec_3_2_A_typical_Supervised_Machine_Learning_Setup_uDcU3ZzH7hs_mp4/20.0.wav'
    audio, sampling_rate = torchaudio.load(file)
    
    outs = model(audio)
    print(outs.shape)
    speaker = outs.cpu().numpy()
    np.save('speaker.npz', speaker)
    # print(torch.sum(outs.view(-1)))


if __name__ == '__main__':
    main()

