from logging import log
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tacotron2Loss(nn.Module):
    def __init__(self, start_iter, total_iter):
        super(Tacotron2Loss, self).__init__()

        self.total_iter = total_iter        
        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()

        self.cur_iter = start_iter

    def forward(self, model_output, targets):
        alpha = round(self.cur_iter / self.total_iter, 2) + 0.05
        beta = 1.0 - alpha
        self.cur_iter += 1

        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out = model_output
        gate_out = gate_out.view(-1, 1)
        
        mel_l2_loss = self.MSE(mel_out, mel_target) + self.MSE(mel_out_postnet, mel_target)
        mel_l1_loss = self.L1(mel_out, mel_target) + self.L1(mel_out_postnet, mel_target)

        mel_loss = beta * mel_l1_loss + alpha * mel_l2_loss
        
        gate_loss = self.BCE(gate_out, gate_target)

        return mel_loss + gate_loss
    

class MiniBatchConstrastiveLoss(nn.Module):
    def __init__(self, t=0.07):
        super(MiniBatchConstrastiveLoss, self).__init__()
        
        self.t = torch.tensor(t).to(device)
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, constrastive_features):
        face_pair_1, face_pair_2 = constrastive_features 
        
        N, C = face_pair_1.shape

        face_pair_1 = F.normalize(face_pair_1, dim=1) 
        face_pair_2 = F.normalize(face_pair_2, dim=1)
        
        logits = torch.matmul(face_pair_1, face_pair_2.t())
        
    
        # positive_pairs, negative_pairs = list(), list()
        # for i in range(N):
        #     for j in range(N):
        #         dot = torch.dot(face_pair_1[i], face_pair_2[j])
        #         if i == j:
        #             positive_pairs.append(dot)
        #         else:
        #             negative_pairs.append(dot)
        # positive_pairs = torch.stack(positive_pairs, dim=0)
        # negative_pairs = torch.stack(negative_pairs, dim=0)
        # print(torch.sum(positive_pairs) - torch.sum(positive_pair))
        # print(torch.sum(negative_pairs) - torch.sum(negative_pair))


        eye = torch.eye(N, dtype=torch.bool).to(device)

        positive_pair = logits[torch.where(eye)]
        negative_pair = logits[torch.where(~eye)]

        postive_labels = torch.ones(N).to(device)
        negative_labels = torch.zeros(N * N - N).to(device)

        loss_pos = self.BCE(positive_pair, postive_labels)
        loss_neg = self.BCE(negative_pair, negative_labels)

        loss = loss_pos + loss_neg

        return loss
