import torch
import torch.nn.functional as F
from torch import nn

try:
    from .backbone import ShuffleNetV2
except:
    from backbone import ShuffleNetV2


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


class VideoExtractor(nn.Module):
    def __init__( self, input_size, in_chns=3):

        super(VideoExtractor, self).__init__()        
        shufflenet = ShuffleNetV2(input_size=input_size, width_mult=1)

        frontend_nout = 24
        self.frontend3D = nn.Sequential(
                                        nn.Conv3d(in_chns, frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                                        nn.BatchNorm3d(frontend_nout),
                                        nn.PReLU(num_parameters=frontend_nout),
                                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                                    )
        
        self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
        
    def forward(self, x):
        B, C, T, H, W = x.size()

        x = self.frontend3D(x)
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        
        x = threeD_to_2D_tensor( x )
        x = self.trunk(x)
        x = x.view(-1, 1024)
        x = x.view(B, Tnew, x.size(1))

        return x


def main():
    model = VideoExtractor(96)
    outputs = model(torch.rand(8, 3, 75, 96, 96))
    print(outputs.shape)


if __name__ == '__main__':
    main()
