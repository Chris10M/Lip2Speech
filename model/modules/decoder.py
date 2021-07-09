import os
from torch import Tensor
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

try:
    from hparams import create_hparams
    from .tacotron2 import Decoder as TacotronDecoder, Postnet, get_mask_from_lengths
except:
    import sys; sys.path.append('../..')
    from hparams import create_hparams
    from tacotron2 import Decoder as TacotronDecoder, Postnet, get_mask_from_lengths


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        print(token_embedding.shape, self.pos_embedding.shape)

        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 n_mels: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.dropout = dropout

        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)

        self.generator = nn.Linear(emb_size, n_mels)

        self.tgt_tok_emb = nn.Linear(n_mels, emb_size)

        self.src_pos_embedding = nn.Parameter(torch.randn(1, src_vocab_size, emb_size))
        self.tgt_pos_embedding = nn.Parameter(torch.randn(1, tgt_vocab_size, emb_size))

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        
        N, T, C = src.shape
        src_emb = F.dropout(src + self.src_pos_embedding[:, :T], self.dropout, self.training)
        
        tgt_emb = self.tgt_tok_emb(trg)
        N, TS, C = tgt_emb.shape
        tgt_emb = F.dropout(tgt_emb + self.tgt_pos_embedding[:, :TS], self.dropout, self.training)
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = torch.max(src).item()
    tgt_seq_len = torch.max(tgt).item()

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)


    N = src.shape[0]

    src_padding_mask = torch.zeros((N, src_seq_len), device=device).type(torch.bool)
    tgt_padding_mask = torch.zeros((N, tgt_seq_len), device=device).type(torch.bool)

    for i in range(N):
        src_padding_mask[i, src[i]:] = True
        tgt_padding_mask[i, tgt[i]:] = True    

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        hparams = create_hparams()

        self.n_mel_channels = hparams.n_mel_channels
        self.mask_padding = hparams.mask_padding

        # self.decoder = TacotronDecoder(hparams)
        # self.postnet = Postnet(hparams)

        self.hparams = hparams

        # state_dict = torch.load('/media/ssd/christen-rnd/Experiments/Lip2Speech/tacotron2_statedict.pt', map_location=device)['state_dict']
        # state_dict.pop('decoder.attention_rnn.weight_ih')
        # state_dict.pop('decoder.attention_layer.memory_layer.linear_layer.weight')
        # state_dict.pop('decoder.decoder_rnn.weight_ih')
        # state_dict.pop('decoder.linear_projection.linear_layer.weight')
        # state_dict.pop('decoder.gate_layer.linear_layer.weight')
        # self.load_state_dict(state_dict, strict=False)


        EMB_SIZE = 1024
        NHEAD = 8
        FFN_HID_DIM = 512
        SRC_VOCAB_SIZE = 30  
        TGT_VOCAB_SIZE = hparams.max_decoder_steps
        NUM_ENCODER_LAYERS = 1
        NUM_DECODER_LAYERS = 4
        N_MELS = hparams.n_mel_channels

        transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, N_MELS, FFN_HID_DIM)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        transformer = transformer.to(device)
        
        self.transformer = transformer


    def forward(self, encoder_outputs, mels, text_lengths, output_lengths):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(text_lengths, output_lengths)        

        mels = mels.permute(0, 2, 1)
        outs = self.transformer(encoder_outputs, mels, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        outs = outs.permute(0, 2, 1)

        return (outs, ) 


        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)
    
    def inference(self, encoder_outputs):
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    
    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs


def main():
    decoder = Decoder()

    visual_features, melspecs, encoder_lengths, melspec_lengths = torch.rand(36, 29, 1024), torch.rand(36, 80, 82), torch.ones(36, dtype=torch.long) * 29, torch.ones(36, dtype=torch.long) * 82
    outs = decoder(visual_features, melspecs, encoder_lengths, melspec_lengths)
    print(outs.shape)

if __name__ == "__main__":
    main()