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
    print(audio.shape)
    outs = model(audio)
    print(outs.shape)
    speaker = outs.cpu().numpy()
    np.save('speaker.npz', speaker)
    # print(torch.sum(outs.view(-1)))


if __name__ == '__main__':
    main()

