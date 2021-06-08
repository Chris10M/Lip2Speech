import cv2
import numpy as np
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch

from vgg_face_recognition.models.vgg_face import VGG_16 as FaceRecognizer



# fr_path = 'vgg_face_recognition/pretrained/vgg_face_torch/VGG_FACE.t7'

# model = FaceRecognizer().double()
# model.load_weights(fr_path)
# im = cv2.imread("vgg_face_recognition/images/ak.png")
# im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()


# model.eval()
# im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
# preds = F.softmax(model(im), dim=1)
# values, indices = preds.max(-1)
# print(model(im).shape)


from transformers import Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import soundfile as sf
import torch

# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

# # load model and tokenizer
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")


"""
/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/hello.mp3
/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/174-84280-0003.flac
/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/6319-275224-0011.flac
/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/174-84280-0002.flac

/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/6319-275224-0011.flac
"""
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

file = '/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/174-84280-0002.flac'
file = '/media/ssd/christen-rnd/Experiments/neural-voice-puppetry/6319-275224-0011.flac'
file = '/media/ssd/christen-rnd/Experiments/Lip2Speech/datasets/avspeech/test/_0kBqBMNfOc_90.000000_97.520000.wav'

'''
Guessed Channel Layout for Input Stream #0.0 : mono
Input #0, wav, from '/media/ssd/christen-rnd/Experiments/Lip2Speech/datasets/avspeech/test/_0kBqBMNfOc_90.000000_97.520000.wav':
  Metadata:
    encoder         : Lavf58.45.100
  Duration: 00:00:07.52, bitrate: 256 kb/s
    Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
'''

# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch

# ds = ds.map(map_to_array)

speech, sampling_rate = sf.read(file, frames=16000 * 3)
assert sampling_rate == 16000
sf.write('test_sound.wav', speech, sampling_rate)

speech = torch.from_numpy(speech.astype(dtype=np.float32)).unsqueeze(0)
speech = torch.cat([speech, speech], 0)

import torch
import fairseq


cp_path = '/media/ssd/christen-rnd/Experiments/Lip2Speech/wav2vec_large.pt'

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

# wav_input_16khz = torch.from_numpy(speech).unsqueeze(0)
z = model.feature_extractor(speech)
c = model.feature_aggregator(z)
print(z.shape, c.shape)

print(sampling_rate)
print(speech.shape)

# output = model(torch.cat([speech.unsqueeze(0), speech.unsqueeze(0)], 0), sampling_rate=sampling_rate, return_tensors='pt')


# print(output.keys())

# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch

# librispeech_eval = librispeech_eval.map(map_to_array)

# def map_to_pred(batch):
#     input_values = tokenizer(batch["speech"], return_tensors="pt", padding="longest").input_values
#     with torch.no_grad():
#         logits = model(input_values.to("cuda")).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = tokenizer.batch_decode(predicted_ids)
#     batch["transcription"] = transcription
#     return batch

# result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

# print("WER:", wer(result["text"], result["transcription"]))



# print(model.config)


# # define function to read in sound file
# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch

# # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)

# # tokenize
# input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1

# # retrieve logits
# logits = model(input_values).logits

# # take argmax and decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = tokenizer.batch_decode(predicted_ids)