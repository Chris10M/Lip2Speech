from logging import log
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

        self.MSE = MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out = model_output
        gate_out = gate_out.view(-1, 1)
        
        mel_loss = self.MSE(mel_out, mel_target) + self.MSE(mel_out_postnet, mel_target)
        gate_loss = self.BCE(gate_out, gate_target)

        return mel_loss + gate_loss
    

class MiniBatchConstrastiveLoss(nn.Module):
    def __init__(self, t=0.07):
        super(MiniBatchConstrastiveLoss, self).__init__()
        
        self.t = torch.tensor(t).to(device)
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, constrastive_features):
        face_features, audio_features = constrastive_features 
        N, C = face_features.shape
        
        F_e = F.normalize(face_features, dim=1) 
        A_e = F.normalize(audio_features, dim=1)

        logits = torch.bmm(A_e.view(N, 1, C), F_e.view(N, C, 1)) * self.t
        logits = logits.squeeze(1).repeat(1, N)

        eye = torch.eye(N, dtype=torch.bool).to(device)
        diag = logits[torch.where(eye)]
        non_diag = logits[torch.where(~eye)]

        postive_labels = torch.ones(N).to(device)
        negative_labels = torch.zeros(N * N - N).to(device)

        loss_pos = self.BCE(diag, postive_labels)
        loss_neg = self.BCE(non_diag, negative_labels)

        loss = loss_pos + loss_neg

        return loss
