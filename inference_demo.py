import torch
import os
import torchaudio

from torch.utils.data import DataLoader

from model import model
from datasets.avspeech import create_hparams, Spec2Audio, AVSpeech


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    model_path = '/media/ssd/christen-rnd/Experiments/Lip2Speech/savedmodels/3694dfbd82dbd1b9d660518e34523228/9000_1624306038.pth'

    hparams = create_hparams()
    spec2audio = Spec2Audio(hparams)

    net = model.get_network('test')
    net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)
    net.eval()

    net = net.to(device)
    
    ds = AVSpeech('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/AVSpeech', mode='test', frame_length=3)
    
    data = ds[1]
    lower_faces, speech, melspec, face_crop = data
    lower_faces = lower_faces.permute(1, 0, 2, 3)

    lower_faces = lower_faces.to(device)
    face_crop = face_crop.to(device)

    mel_outputs, mel_outputs_postnet, _, alignments = net.inference(lower_faces.unsqueeze(0), face_crop.unsqueeze(0))
    speechs = spec2audio(mel_outputs).cpu()

    torchaudio.save('test.wav', spec2audio(melspec), hparams.sampling_rate)
    torchaudio.save('test_recon.wav', speechs, hparams.sampling_rate)



if __name__ == "__main__":
    main()
