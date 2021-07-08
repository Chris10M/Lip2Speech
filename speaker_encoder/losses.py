from jupyterlab_pygments import style
from numpy.core.defchararray import count
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torchvision


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

        losses['l2_loss'] = self.MSE(F.normalize(F.relu(face_embeddings), dim=1),  F.normalize(F.relu(speech_embeddings), dim=1))
        # losses['gram_loss'] = style_criterion(face_embeddings, speech_embeddings)
        
        logits = face_embeddings @ speech_embeddings.T * self.t

        targets = torch.arange(0, N).to(device)
        weight = (torch.ones(N).float() * (N - 1)).to(device)
        
        c_loss = (F.cross_entropy(logits, targets, weight=weight) + F.cross_entropy(logits.T, targets.T, weight=weight)) / 2
        losses['c_loss'] = c_loss


        return losses


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class ReconstuctionLoss(nn.Module):
    def __init__(self):
        super(ReconstuctionLoss, self).__init__()
        
        self.MSE = nn.MSELoss()
        # self.perceptual_loss = VGGPerceptualLoss()
       
    def forward(self, y_pred, y_gt, losses=None):
        if losses is None:
            losses = dict()

        losses['rec_loss'] = 10 * F.mse_loss(y_pred, y_gt)

        # self.perceptual_loss(((y_pred * 128.0) + 127.5), ((y_gt * 128.0) + 127.5))

        return losses

