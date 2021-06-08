import torch
import torch.nn.functional as F
from torch import nn

from attention import MultiHeadAttention

    
# embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)
# # Encoder parameters
# enc_conv_num_layers=3,  # number of encoder convolutional layers
# #enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
# enc_conv_kernel_size = [5,3,3],
# enc_conv_channels=32,  # number of encoder convolutions filters for each layer
# encoder_lstm_units=384,  # number of lstm units for each direction (forward and backward)
# enc_conv_num_blocks=5,
# num_init_filters=24,

class Residual3DBlock(nn.Module):
    def __init__(self):
        super(Residual3DBlock, self).__init__()

        self.conv_block =   nn.Sequential(  
                                            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm3d(32),

                                            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.BatchNorm3d(32),
                                        )

    def forward(self, x):
        residual = x

        return self.conv_block(x) + residual


class LipExtractionModule(nn.Module):
    def __init__(self):
        super(LipExtractionModule, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(24, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(32),            
            nn.ReLU(inplace=True),
        )

        self.layers = nn.Sequential(
            Residual3DBlock(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            Residual3DBlock(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            Residual3DBlock(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            Residual3DBlock(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layers(x)

        x = x.squeeze(3).squeeze(3) 
        # x = [batch, C, T]

        x = x.permute(2, 0, 1)
        # x = [seq_len, batch, input_size]

        return x



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)



class Encoder(nn.Module):
    def __init__(self, emb_dim=32, enc_hid_dim=384, dec_hid_dim=512):
        super().__init__()
        
        self.lip_extractor = LipExtractionModule()
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=2)
        
        self.slf_attn = MultiHeadAttention(n_head=2, d_model=dec_hid_dim, d_k=768, d_v=768)
                
    def forward(self, src):
        
        #src = [src len, batch size]
        
        feature = self.lip_extractor(src)
        
        #embedded = [src len, batch size, emb dim]
        
        encoder_outputs, hidden = self.rnn(feature)
            
        print(encoder_outputs.shape)
        exit(0)
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        print(feature.shape, encoder_outputs.shape)

        enc_output, enc_slf_attn = self.slf_attn(encoder_outputs, encoder_outputs, encoder_outputs, mask=None)

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        # hidden, encoder_outputs
        return 
        return outputs, hidden



def main():
    model = Encoder()
    outputs = model(torch.rand(8, 3, 75, 32, 32))
    print(outputs.shape)

if __name__ == '__main__':
    main()
