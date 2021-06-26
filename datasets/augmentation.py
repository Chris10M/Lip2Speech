import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class FaceAugmentation(nn.Module):
    def __init__(self, p=0.5):
        super(FaceAugmentation, self).__init__()

        self.p = p

    def forward(self, faces):
        if torch.rand(1) < self.p:
            return faces
        
        return [TF.hflip(face) for face in faces]
        

def main():
    import sys;
    sys.path.extend(['./grid', '../model/modules/tacotron2'])
    from torch.utils.data import Dataset, DataLoader
    from grid import GRID, av_speech_collate_fn_pad

    ds = GRID('/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/GRID', mode='test', duration=1, face_augmentation=FaceAugmentation())

    dl = DataLoader(ds,
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=av_speech_collate_fn_pad)

    from IPython.display import Audio, display

    for bdx, batch in enumerate(dl):
        (video, video_lengths), (speeches, audio_lengths), (melspecs, melspec_lengths, mel_gates), faces = batch
        
        frames = video
        print('video.shape', video.shape)
        print('faces.shape ', faces.shape)
        print('frames[0][0].shape ', frames[0][0].shape)
        # print('speech.shape ', speech.shape)

        B, C, T, H, W = video.shape

        for i in range(T):
            image = frames[0, :, i, :, :].permute(1, 2, 0).numpy()
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            
            print(i, image.shape)


            cv2.imshow('image', image[:, :, :: -1])

            if ord('q') == cv2.waitKey(0):
                exit()


if __name__ == '__main__':
    main()
