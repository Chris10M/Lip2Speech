import torch
import torch.nn.functional as F
from torch import nn

# from attention import MultiHeadAttention

try:
    from .video import VideoExtractor
except:
    from video import VideoExtractor

    
# embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)
# # Encoder parameters
# enc_conv_num_layers=3,  # number of encoder convolutional layers
# #enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
# enc_conv_kernel_size = [5,3,3],
# enc_conv_channels=32,  # number of encoder convolutions filters for each layer
# encoder_lstm_units=384,  # number of lstm units for each direction (forward and backward)
# enc_conv_num_blocks=5,
# num_init_filters=24,


class Encoder(nn.Module):
    def __init__(self, enc_hid_dim=512, dec_hid_dim=512):
        super(Encoder, self).__init__()

        self.feature_extractor = VideoExtractor(input_size=96)
        self.rnn = nn.GRU(1024, enc_hid_dim, bidirectional=True, num_layers=2)


    def forward(self, x):
        x = self.feature_extractor(x)
        # x = [batch, C, T]

        embeddings = x.permute(1, 0, 2)

        # seq_len, batch, input_size
        encoder_outputs, hidden = self.rnn(embeddings)
        print(encoder_outputs.shape)
        exit(0)

        return encoder_outputs



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



# class Encoder(nn.Module):
#     def __init__(self, emb_dim=32, enc_hid_dim=384, dec_hid_dim=512):
#         super().__init__()
        
#         self.lip_extractor = LipExtractionModule()
#         self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=2)
        
#         self.slf_attn = MultiHeadAttention(n_head=2, d_model=dec_hid_dim, d_k=768, d_v=768)
                
#     def forward(self, src):
        
#         #src = [src len, batch size]
        
#         feature = self.lip_extractor(src)
        
#         #embedded = [src len, batch size, emb dim]
        
#         encoder_outputs, hidden = self.rnn(feature)
            
#         print(encoder_outputs.shape)
#         exit(0)
#         #outputs = [src len, batch size, hid dim * num directions]
#         #hidden = [n layers * num directions, batch size, hid dim]
        
#         #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
#         #outputs are always from the last layer
        
#         #hidden [-2, :, : ] is the last of the forwards RNN 
#         #hidden [-1, :, : ] is the last of the backwards RNN
        
#         #initial decoder hidden is final hidden state of the forwards and backwards 
#         #  encoder RNNs fed through a linear layer
#         # hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
#         print(feature.shape, encoder_outputs.shape)

#         enc_output, enc_slf_attn = self.slf_attn(encoder_outputs, encoder_outputs, encoder_outputs, mask=None)

#         #outputs = [src len, batch size, enc hid dim * 2]
#         #hidden = [batch size, dec hid dim]
        
#         # hidden, encoder_outputs
#         return 
#         return outputs, hidden



def main():
    model = Encoder()
    outputs = model(torch.rand(8, 3, 75, 96, 96))
    print(outputs.shape)

if __name__ == '__main__':
    main()
