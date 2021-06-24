import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .tacotron2 import Decoder as TacotronDecoder, Postnet, create_hparams, get_mask_from_lengths
except:
    from tacotron2 import Decoder as TacotronDecoder, Postnet, create_hparams, get_mask_from_lengths


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        hparams = create_hparams()
        hparams.n_mel_channels = hparams.filter_length // 2 + 1 # Linear Spectogram usage

        self.n_mel_channels = hparams.n_mel_channels
        self.mask_padding = hparams.mask_padding

        self.decoder = TacotronDecoder(hparams)
        self.postnet = Postnet(hparams)

        self.hparams = hparams

    def forward(self, encoder_outputs, mels, text_lengths, output_lengths):
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

    decoder(torch.rand(4, 37, 1536))


if __name__ == "__main__":
    main()