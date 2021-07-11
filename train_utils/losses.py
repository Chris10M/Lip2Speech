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
        
        mel_l1_loss = self.L1(mel_out, mel_target)
        mel_loss = self.MSE(mel_out, mel_target) 
        postnet_loss = self.MSE(mel_out_postnet, mel_target)

        gate_loss = self.BCE(gate_out, gate_target)

        return mel_loss, postnet_loss, gate_loss, mel_l1_loss     


class TLoss(nn.Module):
    def __init__(self):
        super(TLoss, self).__init__()

        self.BCE = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()

    def forward(self, model_output, targets, losses=None):
        if losses is None:
            losses = dict()

        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out = model_output[0]
        mel_out_postnet = model_output[1]
        gate_out = model_output[2]

        gate_out = gate_out.view(-1, 1)
        
        # print(mel_out.shape, mel_target.shape)

        losses['mel_loss'] = self.MSE(mel_out, mel_target) 
        losses['postnet_mel_loss'] = self.MSE(mel_out_postnet, mel_target)
        losses['gate_loss'] = self.BCE(gate_out, gate_target)

        return losses     



class MiniBatchConstrastiveLoss(nn.Module):
    def __init__(self, t=0.07):
        super(MiniBatchConstrastiveLoss, self).__init__()
        
        self.t = torch.tensor(t).to(device)
        self.BCE = nn.BCEWithLogitsLoss()
       
    def forward(self, constrastive_features):
        face_pair_1, face_pair_2, encoded_mel_gt, encoded_mel_pred = constrastive_features 
        
        l1_loss = self.L1(encoded_mel_gt, face_embeddings) + self.L1(encoded_mel_pred, face_embeddings)
        
        face_embeddings_norm = F.normalize(face_embeddings, dim=1)
        l2_loss = self.MSE(F.normalize(encoded_mel_gt, dim=1), face_embeddings_norm) + self.MSE(F.normalize(encoded_mel_pred, dim=1), face_embeddings_norm)
        
        return l1_loss + 0.025 * l2_loss
        
        N, C = face_pair_1.shape

        face_pair_1 = F.normalize(face_pair_1, dim=1) 
        face_pair_2 = F.normalize(face_pair_2, dim=1)
        
        logits = torch.matmul(face_pair_1, face_pair_2.t())
           
        # # print(torch.cdist(face_pair_1, face_pair_2))
        # # positive_pairs, negative_pairs = list(), list()
        # # for i in range(N):
        # #     for j in range(N):
        # #         dot = torch.dot(face_pair_1[i], face_pair_2[j])
        # #         if i == j:
        # #             positive_pairs.append(dot)
        # #         else:
        # #             negative_pairs.append(dot)
        # # positive_pairs = torch.stack(positive_pairs, dim=0)
        # # negative_pairs = torch.stack(negative_pairs, dim=0)
        # # print(positive_pairs, negative_pairs)
        # # print(torch.sum(positive_pairs) - torch.sum(positive_pair))
        # # # print(torch.sum(negative_pairs) - torch.sum(negative_pair))


        eye = torch.eye(N, dtype=torch.bool).to(device)

        positive_pair = logits[torch.where(eye)]
        negative_pair = logits[torch.where(~eye)]
        

        postive_labels = torch.ones(N).to(device)
        negative_labels = torch.zeros(N * N - N).to(device)

        loss_pos = self.BCE(positive_pair, postive_labels)
        loss_neg = self.BCE(negative_pair, negative_labels)

        loss = loss_pos + loss_neg

        return loss
