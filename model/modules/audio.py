import torch
import torch.nn.functional as F
from torch import nn
import fairseq
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import T

try:
    from tacotron2.hparams import create_hparams
except:
    from .tacotron2.hparams import create_hparams


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
    def __init__(self):
        super().__init__()
        
        hparams  = create_hparams()
        in_chns = hparams.filter_length // 2 + 1

        self.lstm = nn.LSTM(input_size=in_chns,
                            hidden_size=256, 
                            num_layers=3, 
                            batch_first=True)

        self.linear = nn.Linear(in_features=256, out_features=256)
        
        self.projection = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=512),
        )

        state_dict = torch.load('/media/ssd/christen-rnd/Experiments/Lip2Speech/speaker_encoder.pt')['model_state']
        state_dict.pop('lstm.weight_ih_l0')
        
        self.load_state_dict(state_dict, strict=False)
        for name, p in self.named_parameters():
            if 'projection' in name: continue
            if 'l0' in name: continue 

            p.requires_grad_(False)

    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        embeds_raw = self.linear(hidden[-1])
        embeds_raw = self.projection(embeds_raw)

        embeds = F.normalize(embeds_raw, dim=1, p=2)

        return embeds



def main():
    # torch.Size([1, 513, 1000])

    # model = AudioExtractor('/media/ssd/christen-rnd/Experiments/Lip2Speech/wav2vec_large.pt')
    # speech = torch.rand(1, 16000 * 1)
    # speech = torch.cat([speech, speech], 0)

    # outputs = model.identity_features(speech)
    # print(outputs.shape)

    model = SpeakerEncoder()
    outs = model(torch.rand(2, 1000, 513))

    print(outs.shape)


if __name__ == '__main__':
    main()

