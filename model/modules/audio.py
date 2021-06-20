import torch
import torch.nn.functional as F
from torch import nn
import fairseq
from torch.nn.modules.container import T


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
            nn.Dropout(0.3),
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



def main():
    model = AudioExtractor('/media/ssd/christen-rnd/Experiments/Lip2Speech/wav2vec_large.pt')
    speech = torch.rand(1, 16000 * 1)
    speech = torch.cat([speech, speech], 0)

    outputs = model.identity_features(speech)
    print(outputs.shape)


if __name__ == '__main__':
    main()

