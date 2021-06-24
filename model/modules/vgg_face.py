import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = InceptionResnetV1(pretrained='casia-webface')
        for p in self.resnet.parameters():
            p.requires_grad_(False)

        self.projection_layer = nn.Sequential(            
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        embeddings = self.resnet(x)
        projection = self.projection_layer(embeddings)

        return projection


def main():
    from facenet_pytorch import MTCNN, InceptionResnetV1

    
    mtcnn = MTCNN(
                    image_size=160, margin=0, min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                    device=device
    )

    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    
    x = cv2.imread("/media/ssd/christen-rnd/Experiments/Lip2Speech/vgg_face_recognition/images/ak.png")[:, :, ::-1]

    x_aligned, prob = mtcnn(x, return_prob=True)

    aligned = x_aligned.unsqueeze(0)
    e1 = resnet(aligned).detach().cpu()
    
    mtcnn(x, 'test.jpg', return_prob=True)

    im = cv2.imread("test.jpg")[:, :, ::-1].copy()
    im = torch.tensor(im).permute(2, 0, 1)
    
    x_aligned = (im.float() - 127.5) / 128.0  


    aligned = x_aligned.unsqueeze(0)
    e2 = resnet(aligned).detach().cpu()
    print(e1.shape)
    print((e2 - e1).norm())


if __name__ == "__main__":
    main()