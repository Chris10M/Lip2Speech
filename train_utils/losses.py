from logging import log
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

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

        losses['mel_loss'] = 10 * self.MSE(mel_out, mel_target) 
        losses['postnet_mel_loss'] = self.MSE(mel_out_postnet, mel_target)
        losses['gate_loss'] = self.BCE(gate_out, gate_target)

        return losses     



class AdversarialLoss(nn.Module):
    def __init__(self, optim_D):
        super(AdversarialLoss, self).__init__()
        
        self.optimizer_D = optim_D
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, D, model_output, face_features, losses=None):
        if losses is None:
            losses = dict()

        
        real, fake = model_output
        face_features = face_features.detach()

        real = real.detach()        
        fake = fake

        real_pred, real_pred_features = D(real, face_features, False, True) # False - generate random patch only once, and keep it accross all loss computation.
        fake_pred, fake_pred_features = D(fake, face_features, True, True)


        d_fm_loss = 0
        for real_features, fake_features in zip(real_pred_features, fake_pred_features):
            d_fm_loss += F.l1_loss(fake_features.view(-1), real_features.detach().view(-1))                                        

        losses['g_loss'] = -torch.mean(fake_pred)
        losses['g_d_fm_loss'] = 10 * d_fm_loss

        return losses

    def discriminator_forward(self, D, model_output, face_features, losses=None):
        self.optimizer_D.zero_grad()

        real, fake = model_output
        face_features = face_features.detach() 

        real = real.detach()
        fake = fake.detach()

        # lambda_gp = 10
        # gradient_penalty = self.compute_gradient_penalty(D, real.data, fake.data, face_features)
        d_loss = -torch.mean(D(real, face_features, True)) + torch.mean(D(fake, face_features, True)) #+ lambda_gp * gradient_penalty


        losses['d_loss'] = d_loss

        d_loss.backward()
        self.optimizer_D.step()

        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        return losses

    def compute_gradient_penalty(self, D, real_samples, fake_samples, face_features):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(torch.rand(real_samples.size(0), 1, 1)).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = D(interpolates, face_features, True).reshape(real_samples.size(0), 1)
        fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).to(device).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
