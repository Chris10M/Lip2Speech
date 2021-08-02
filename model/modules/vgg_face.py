import cv2
from matplotlib.pyplot import winter
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import torchaudio.transforms as AT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = InceptionResnetV1(pretrained='casia-webface')
        for p in self.resnet.parameters():
            p.requires_grad_(False)
        self.resnet.last_linear.requires_grad_(True)
        self.resnet.last_bn.requires_grad_(True)
        
        self.projection_layer = nn.Sequential(  
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        x = self.resnet.conv2d_1a(x)
        x = self.resnet.conv2d_2a(x)
        x = self.resnet.conv2d_2b(x)
        x = self.resnet.maxpool_3a(x)
        x = self.resnet.conv2d_3b(x)
        x = self.resnet.conv2d_4a(x)
        x = self.resnet.conv2d_4b(x)
        x = self.resnet.repeat_1(x)
        x = self.resnet.mixed_6a(x)
        x = self.resnet.repeat_2(x)
        x = self.resnet.mixed_7a(x)
        x = self.resnet.repeat_3(x)
        x = self.resnet.block8(x)
        x = self.resnet.avgpool_1a(x)
        x = self.resnet.dropout(x)
        x = self.resnet.last_linear(x.view(x.shape[0], -1))
        embeddings_raw = self.resnet.last_bn(x)

        # projection = F.relu(self.projection_layer(embeddings_raw))
        projection = (self.projection_layer(embeddings_raw))

        return projection

    def inference(self, x):
        if self.training:
            self = self.eval()
        
        with torch.no_grad():
            embeds_raw = self(x)
            embeds_raw = F.relu(embeds_raw)

        return F.normalize(embeds_raw, p=2, dim=1)


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