from .face_utils import align_and_crop_face
from .spectograms import MelSpectrogram, MelSpec2Audio
from .augmentation import *
import torch


def train_collate_fn_pad(batch):
    lower_faces, speeches, melspecs, face_crop = zip(*batch)
    
    max_frames_in_batch = max([l.shape[0] for l in lower_faces])
    max_samples_in_batch = max([s.shape[1] for s in speeches])
    max_melspec_samples_in_batch = max([m.shape[1] for m in melspecs])

    padded_lower_faces = torch.zeros(len(lower_faces), max_frames_in_batch, *tuple(lower_faces[0].shape[1:]))
    padded_speeches = torch.zeros(len(speeches), 1, max_samples_in_batch)
    padded_melspecs = torch.ones(len(melspecs), melspecs[0].shape[0], max_melspec_samples_in_batch) * -11.5129
    mel_gate_padded = torch.zeros(len(melspecs), max_melspec_samples_in_batch)

    video_lengths = list()
    audio_lengths = list()
    melspec_lengths = list()
    for idx, (lower_face, speech, melspec) in enumerate(zip(lower_faces, speeches, melspecs)):
        T = lower_face.shape[0]
        video_lengths.append(T)

        padded_lower_faces[idx, :T, :, :, :] = lower_face

        S = speech.shape[-1]
        audio_lengths.append(S)
        padded_speeches[idx, :, :S] = speech
        
        M = melspec.shape[-1]
        melspec_lengths.append(M)
        padded_melspecs[idx, :, :M] = melspec

        mel_gate_padded[idx, M-1:] = 1.0

    face_crop_tensor = torch.cat([f.unsqueeze(0) for f in face_crop], dim=0)
    padded_lower_faces = padded_lower_faces.permute(0, 2, 1, 3, 4)
    padded_speeches = padded_speeches.squeeze(1) 

    video_lengths = torch.tensor(video_lengths)
    audio_lengths = torch.tensor(audio_lengths)
    melspec_lengths = torch.tensor(melspec_lengths)

    return (padded_lower_faces, video_lengths), (padded_speeches, audio_lengths), (padded_melspecs, melspec_lengths, mel_gate_padded), face_crop_tensor



def test_collate_fn_pad(batch):
    lower_faces, speeches, melspecs, face_crop, file_paths = zip(*batch)
    
    max_frames_in_batch = max([l.shape[0] for l in lower_faces])
    max_samples_in_batch = max([s.shape[1] for s in speeches])
    max_melspec_samples_in_batch = max([m.shape[1] for m in melspecs])

    padded_lower_faces = torch.zeros(len(lower_faces), max_frames_in_batch, *tuple(lower_faces[0].shape[1:]))
    padded_speeches = torch.zeros(len(speeches), 1, max_samples_in_batch)
    padded_melspecs = torch.ones(len(melspecs), melspecs[0].shape[0], max_melspec_samples_in_batch) * -11.5129
    mel_gate_padded = torch.zeros(len(melspecs), max_melspec_samples_in_batch)

    video_lengths = list()
    audio_lengths = list()
    melspec_lengths = list()
    for idx, (lower_face, speech, melspec) in enumerate(zip(lower_faces, speeches, melspecs)):
        T = lower_face.shape[0]
        video_lengths.append(T)

        padded_lower_faces[idx, :T, :, :, :] = lower_face

        S = speech.shape[-1]
        audio_lengths.append(S)
        padded_speeches[idx, :, :S] = speech
        
        M = melspec.shape[-1]
        melspec_lengths.append(M)
        padded_melspecs[idx, :, :M] = melspec

        mel_gate_padded[idx, M-1:] = 1.0

    face_crop_tensor = torch.cat([f.unsqueeze(0) for f in face_crop], dim=0)
    padded_lower_faces = padded_lower_faces.permute(0, 2, 1, 3, 4)
    padded_speeches = padded_speeches.squeeze(1) 

    video_lengths = torch.tensor(video_lengths)
    audio_lengths = torch.tensor(audio_lengths)
    melspec_lengths = torch.tensor(melspec_lengths)

    return (padded_lower_faces, video_lengths), (padded_speeches, audio_lengths), (padded_melspecs, melspec_lengths, mel_gate_padded), face_crop_tensor, file_paths
