from numpy.core.defchararray import decode
import torch
import torchaudio.transforms as AT
import torch.nn.functional as F
from torch import nn
import numpy as np
from facenet_pytorch import InceptionResnetV1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class FaceRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = InceptionResnetV1(pretrained='casia-webface')
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.resnet.last_linear.requires_grad_(True)
        self.resnet.last_bn.requires_grad_(True)
        
        self.projection_layer = nn.Sequential(  
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        x = self.resnet.conv2d_1a(x)
        x = self.resnet.conv2d_2a(x)
        x = self.resnet.conv2d_2b(x)
        x = self.resnet.maxpool_3a(x)
        x = self.resnet.conv2d_3b(x)
        x = self.resnet.conv2d_4a(x)
        x = self.resnet.conv2d_4b(x)
        x = self.resnet.repeat_1(x)
        x = self.resnet.mixed_6a(x)
        x = self.resnet.repeat_2(x)
        x = self.resnet.mixed_7a(x)
        x = self.resnet.repeat_3(x)
        x = self.resnet.block8(x)
        x = self.resnet.avgpool_1a(x)
        x = self.resnet.dropout(x)
        x = self.resnet.last_linear(x.view(x.shape[0], -1))
        embeddings_raw = self.resnet.last_bn(x)

        # projection = F.relu(self.projection_layer(embeddings_raw))
        projection = (self.projection_layer(embeddings_raw))

        return projection

    def inference(self, x):
        embeds_raw = self(x)
        embeds_raw = F.relu(embeds_raw)

        return F.normalize(embeds_raw, p=2, dim=1)


class SpeakerEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=40,
                            hidden_size=256, 
                            num_layers=3, 
                            batch_first=True)

        self.linear = nn.Linear(in_features=256, out_features=256)
        
        state_dict = torch.load('/media/ssd/christen-rnd/Experiments/Lip2Speech/speaker_encoder.pt', map_location=device)['model_state']        
        self.load_state_dict(state_dict, strict=False)

        for name, p in self.named_parameters():
            p.requires_grad_(False)

        self.mel_spec = AT.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40).to(device)


    def forward(self, utterances, hidden_init=None):
        utterances = self.mel_spec(utterances).permute(0, 2, 1)
        
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # embeds_raw = F.relu(self.linear(hidden[-1]))
        embeds_raw = (self.linear(hidden[-1]))

        # embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds_raw
        
    def inference(self, x):
        embeds_raw = F.relu(self(x))
        return F.normalize(embeds_raw, p=2, dim=1)



class SpeakerDecoder(nn.Module):        
    def __init__(self):
        super().__init__()

        self.seq_len = 201
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256, 
                            num_layers=3, 
                            batch_first=True)
        self.linear = nn.Linear(256, 40)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state)  = self.lstm(x)
    
        x = self.linear(x)
                
        return x



class ConvBlock(nn.Module):
    def conv3x3(self, in_chns, out_chns, exp_r=6):
        return nn.Sequential(
            nn.Conv2d(in_chns, in_chns * exp_r, 1),
            nn.BatchNorm2d(in_chns * exp_r),
            nn.ReLU(True),

            nn.Conv2d(in_chns * exp_r, out_chns, 3, groups=out_chns, padding=3 // 2),
            nn.BatchNorm2d(out_chns),
            nn.ReLU(True),
        )

    def __init__(self, in_chns, out_chns):
        super().__init__()
        self.conv = self.conv3x3(in_chns, out_chns)
        self.upsample = nn.Conv2d(in_chns, out_chns, 1) 

    def forward(self, x):
        residual = x
        return self.conv(x) + self.upsample(residual) 


class FaceDecoder(nn.Module):        
        
    def upsample(self, chns, scale):
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale),
            nn.Conv2d(chns, chns, 3, padding=3 // 2)
        )

    def __init__(self):
        super().__init__()

        self.res = 5
        self.lsize = 512
        self.linear = nn.Linear(256, self.res * self.res * self.lsize)
        self.feature_extractor = nn.Sequential(
            ConvBlock(512, 256),
            self.upsample(256, 2),

            ConvBlock(256, 128),
            self.upsample(128, 2),

            ConvBlock(128, 64),
            self.upsample(64, 2),

            ConvBlock(64, 64),
            self.upsample(64, 2),

            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):    
        x = self.linear(x)

        x = F.dropout(x, 0.3, self.training)
        x = x.view(-1, self.lsize, self.res, self.res)

        x = self.feature_extractor(x)

        x = F.interpolate(x, (160, 160), mode='bilinear', align_corners=True)

        return x


def get_network():
    fnet = FaceRecognizer().to(device)
    fnet = fnet.train()
    
    snet = SpeakerEncoder(device).to(device)
    snet = snet.eval()

    return fnet, snet


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
        
    model = SpeakerEncoder('cpu')

    file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/DL/Deep_Learning(CS7015)___Lec_3_2_A_typical_Supervised_Machine_Learning_Setup_uDcU3ZzH7hs_mp4/20.0.wav'
    audio, sampling_rate = torchaudio.load(file)
    audio = audio[:, :16000]
    melspec = AT.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40).to(device)
    utterances = melspec(audio).permute(0, 2, 1)
    print(audio.shape, utterances.shape)

    outs = model(audio)
    print(outs.shape)
    speaker = outs.cpu().numpy()
    # np.save('speaker.npz', speaker)
    # print(torch.sum(outs.view(-1)))
    
    decoder = SpeakerDecoder()
    decode = decoder(torch.rand(2, 256))
    print('speakerdecode', decode.shape)

    decoder = FaceDecoder()
    decode = decoder(torch.rand(2, 256))
    print('facedecode', decode.shape)
    

if __name__ == '__main__':
    main()

