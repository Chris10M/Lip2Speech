from numpy import random
from numpy.lib.npyio import save
import torch
from model import FaceRecognizer, SpeakerEncoder
import numpy as np
import cv2

from dataset import AVSpeechFace, av_speech_face_collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = AVSpeechFace('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='train')

saved_path = '/media/ssd/christen-rnd/Experiments/Lip2Speech/speaker_encoder/savedmodels/156bcbfe7c66281240affb1d053dd279/249000_1625097402.pth'
saved_path = 'savedmodels/156bcbfe7c66281240affb1d053dd279/333000_1625226579.pth'

snet = SpeakerEncoder(device).to(device)
snet = snet.eval()

net = FaceRecognizer().to(device)
net = net.eval()


loaded_model = torch.load(saved_path, map_location=device)
state_dict = loaded_model['state_dict']
net.load_state_dict(state_dict, strict=False)

for i in range(len(ds)):
    speech, face = ds[i]


    print(speech.shape, face.shape)

    with torch.no_grad():
        speech_embeds = snet.inference(speech).cpu().numpy()[0]
        face_embeds = net.inference(face.unsqueeze(0)).cpu().numpy()[0]

    cv_face_image = ((face.permute(1, 2, 0).numpy() * 128.0) + 127.5).astype(dtype=np.uint8)[:, :, :: -1]
    
    cv2.imwrite('face.jpg', cv_face_image)
    np.save('speech_embeds', speech_embeds)
    np.save('face_embeds', face_embeds)

    cv2.imshow('cv_face_image', cv_face_image)
    if ord('q') == cv2.waitKey(0):
        exit()
