from jupyterlab_pygments import style
from numpy.core.defchararray import count
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gram_matrix(features):
    N, C = features.shape  
        
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(C)


def style_criterion(logits, target):
    return F.mse_loss(gram_matrix(logits), gram_matrix(target).detach())



class MiniBatchConstrastiveLoss(nn.Module):
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
            
    def __init__(self, t=1.0):
        super(MiniBatchConstrastiveLoss, self).__init__()
        
        self.t = nn.Parameter(torch.tensor(t, requires_grad=True, device=device))
        self.BCE = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()
        self.count = 0
       
    def forward(self, constrastive_features, losses=None):
        self.t.data = self.t.data.clamp(max=100)

        if losses is None:
            losses = dict()

        speech_embeddings, face_embeddings = constrastive_features 
        N = face_embeddings.shape[0]

        # losses['l2_loss'] = self.MSE(F.normalize(F.relu(face_embeddings), dim=1),  F.normalize(F.relu(speech_embeddings), dim=1))
        # losses['gram_loss'] = style_criterion(face_embeddings, speech_embeddings)
        
        logits = face_embeddings @ speech_embeddings.T * self.t
        # if self.count % 1000 == 0:
        #     print(logits)
        #     print(torch.argmax(F.softmax(logits, dim=-1), dim=0), torch.argmax(F.softmax(logits, dim=-1), dim=1))

        # self.count += 1
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

        # face_similarity = face_embeddings @ face_embeddings.T
        # speech_similarity = speech_embeddings @ speech_embeddings.T
        
        # targets = F.softmax((face_similarity + speech_similarity) / 2, dim=-1)
        targets = torch.arange(0, N).to(device)#F.softmax(speech_similarity, dim=-1)
        
        c_loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets.T)) / 2
        losses['c_loss'] = c_loss
        # c_loss = (c_loss / 2).mean()

        # eye = torch.eye(N, dtype=torch.bool).to(device)

        # positive_pair = logits[torch.where(eye)]
        # negative_pair = logits[torch.where(~eye)]
                
        # postive_labels = torch.ones(N).to(device)
        # negative_labels = torch.zeros(N * (N - 1)).to(device)

        # loss_pos = F.binary_cross_entropy_with_logits(positive_pair, postive_labels, torch.ones(N).to(device) * (N - 1))  # self.BCE(positive_pair, postive_labels)
        # loss_neg = F.binary_cross_entropy_with_logits(negative_pair, negative_labels)

        # c_loss = loss_pos + loss_neg


        return losses

