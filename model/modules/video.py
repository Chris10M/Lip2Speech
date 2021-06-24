from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
import torch
import torch.nn.functional as F
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.size()

        x = x.permute(0, 2, 1, 3, 4)

        x_reshape = x.contiguous().view(-1, C, H, W) 
    
        y = self.module(x_reshape)
        
        BT, C, H, W  = y.shape
        y = y.contiguous().view(B, T, C, H, W) 

        y = y.permute(0, 2, 1, 3, 4)

        return y


class VideoExtractor(nn.Module):
    def __init__( self, input_size, in_chns=3):

        super(VideoExtractor, self).__init__()        
        shufflenet = shufflenet_v2_x1_0(pretrained=True)

        frontend_nout = 24
        self.frontend3D = nn.Sequential(
                                        nn.Conv3d(in_chns, frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                                        nn.BatchNorm3d(frontend_nout),
                                        nn.PReLU(num_parameters=frontend_nout),
                                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                                    )
                
        self.trunk = nn.Sequential(shufflenet.stage2,
                                   shufflenet.stage3,
                                   shufflenet.stage4,
                                   shufflenet.conv5,
                                   nn.AdaptiveAvgPool2d(1))
                                   
        self.backend1D = nn.Sequential(
            nn.Conv1d(1024, 1024, 5, 1, 5 // 2),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Conv1d(1024, 1024, 5, 1, 5 // 2),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.time_distributed = TimeDistributed(self.trunk)

    def forward(self, x):
        x = self.frontend3D(x)
        
        x = self.time_distributed(x)

        B, C, T, H, W = x.shape
        x = x.view(B, C, T)

        x = self.backend1D(x).permute(0, 2, 1)

        return x


def main():
    from torchstat import stat

    model = VideoExtractor(96)
    model.eval()
    
    inp = torch.rand(5, 3, 30, 96, 96)

    outputs = model(inp)
    # print(outputs[:, 0, :8])
    print(outputs.shape)
    
    # outputs = model(inp[:, :, :6, :, :])
    # print(outputs[:, 0, :8])

    # print(outputs.shape)


if __name__ == '__main__':
    main()
