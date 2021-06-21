from soundfile import SoundFile
import torchaudio
from logger import setup_logger
import torch.multiprocessing as mp
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import hashlib
import os
import os.path as osp
import logging
import time
import datetime
import argparse
import imutils

from datasets.avspeech.dataset import AVSpeech, av_speech_collate_fn_pad
from train_utils.optimizer import Optimzer
from train_utils.losses import *
from model import model
from model.modules.tacotron2.hparams import create_hparams
# from evaluate import evaluate_net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Logger:
    logger = None
    ModelSavePath = 'model'


def set_model_logger(net):
    model_info = str(net)

    respth = f'savedmodels/{hashlib.md5(model_info.encode()).hexdigest()}'
    Logger.ModelSavePath = respth

    if not osp.exists(respth): os.makedirs(respth)
    logger = logging.getLogger()

    if setup_logger(respth):
        logger.info(model_info)

    Logger.logger = logger


def main():
    hparams = create_hparams()
    
    ds = AVSpeech('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='test')

    net = model.get_network('train').to(device)
    set_model_logger(net)
    
    saved_path = ''
    
    max_iter = 64000
    save_iter = 1000
    n_img_per_gpu = 6
    n_workers = min(n_img_per_gpu, 16)
    
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=True,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True, 
                    collate_fn=av_speech_collate_fn_pad)


    reconstruction_criterion = Tacotron2Loss()
    contrastive_criterion = MiniBatchConstrastiveLoss()
    
    optim = Optimzer(net, 0, max_iter, weight_decay=hparams.weight_decay)

    min_eval_loss = 1e5
    epoch = 0
    start_it = 0
    if os.path.isfile(saved_path):
        loaded_model = torch.load(saved_path)
        state_dict = loaded_model['state_dict']

        try:
            net.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(e)
        
        try:
            start_it = 0
            start_it = loaded_model['start_it'] + 2
        except KeyError:
            start_it = 0

        try:
            epoch = loaded_model['epoch']
        except KeyError:
            epoch = 0

        try:
            min_eval_loss = loaded_model['min_eval_loss']
        except KeyError: ...

        try:
            optim = Optimzer(net, start_it, max_iter)
            optim.load_state_dict(loaded_model['optimize_state'])
            ...
        except (ValueError, KeyError): pass


        print(f'Model Loaded: {saved_path} @ start_it: {start_it}')

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)

    # start_training = False
    for it in range(start_it, max_iter):
        try:
            batch = next(diter)
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            
            batch = next(diter)

        (videos, video_lengths), (audios, audio_lengths), (melspecs, melspec_lengths, mel_gates), face_crops = batch

        videos, audios, melspecs, face_crops = videos.to(device), audios.to(device), melspecs.to(device), face_crops.to(device)
        video_lengths, audio_lengths, melspec_lengths = video_lengths.to(device), audio_lengths.to(device), melspec_lengths.to(device)
        mel_gates = mel_gates.to(device)
        outputs, constrastive_features = net(videos, face_crops, audios, melspecs, video_lengths, audio_lengths, melspec_lengths)
        
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
    
        loss = reconstruction_criterion((mel_outputs, mel_outputs_postnet, gate_outputs), (melspecs, mel_gates))
        loss += contrastive_criterion(constrastive_features)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), hparams.grad_clip_thresh)

            # optim.update_lr()

        optim.step()
        optim.zero_grad()

        loss_avg.append(loss.item())

        print(it)

        if (it + 1) % save_iter == 0 or os.path.isfile('save'):
            save_pth = osp.join(Logger.ModelSavePath, f'{it + 1}_{int(time.time())}.pth')

            evaluation = evaluate_net(args, net)
            Logger.logger.info(f"Model@{it + 1}\n{evaluation}")
            
            eval_loss = evaluation.loss()
            optim.reduce_lr_on_plateau(eval_loss)

            if eval_loss < min_eval_loss:  
                print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
                torch.save({
                    'epoch': epoch,
                    'start_it': it,
                    'state_dict': net.state_dict(),
                    'optimize_state': optim.state_dict(),
                    'min_eval_loss': min_eval_loss,
                }, save_pth)
                print(f'model at: {(it + 1)} Saved')

                min_eval_loss = eval_loss

        #   print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    f'epoch: {epoch}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            Logger.logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
    net.cpu()
    torch.save({'state_dict': net.state_dict()}, save_pth)

    Logger.logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    main()
