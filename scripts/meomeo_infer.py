from mimitransformers import TrainingMimiModel, TrainingMimiProjectorModel
from mimitrainer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
from mimitrainer.trainer import MimiTrainer
from pathlib import Path
from transformers import MimiModel, AutoFeatureExtractor
import torchaudio
from torchaudio.transforms import Resample
import torch
import numpy as np

# Instantiate model and feature extractor
meomeo = TrainingMimiProjectorModel.from_pretrained("phongtintruong/meomeo-kaggle")
mimi = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

def load_wav(wav_path, sampling_rate):
  waveform, sample_rate = torchaudio.load(wav_path)
  print(waveform.shape)
  if sample_rate != sampling_rate:
      resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
      waveform = resampler(waveform)

  wav_audio_sample = waveform.squeeze().numpy()
  return wav_audio_sample


def enc_dec(audio_sample, model, feature_extractor, num_quantizers=32):
  tensor_audio = torch.tensor(audio_sample).unsqueeze(0).unsqueeze(0)
  print(tensor_audio)
  print(tensor_audio.shape)
  print('-'*100)
  # pre-process the inputs
  inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
  print(inputs['input_values'])
  print(inputs['input_values'].shape)
  print('-'*100)
  # Check if they are equal
  if torch.equal(tensor_audio, inputs['input_values']):
    print("The tensors are equal.")
  else:
    print("The tensors are not equal.")
  # explicitly encode then decode the audio inputs
  # encoder_outputs = model.encode(inputs["input_values"], num_quantizers=num_quantizers)
  # audio_values = model.decode(encoder_outputs.audio_codes)[0].detach().numpy().flatten()

  # or the equivalent with a forward pass
  model_out = model(inputs["input_values"], num_quantizers=num_quantizers)
  print(model_out)
  audio_values = model_out.audio_values
  audio_codes = model_out.audio_codes
  print(audio_values.shape)
  print(audio_codes.shape)
  return audio_values

def save_wav(audio_sample, save_path, sampling_rate):
    # Convert the audio sample to a tensor (if it is numpy array)
    if isinstance(audio_sample, np.ndarray):
        audio_sample = torch.tensor(audio_sample)
    
    # Ensure it's a 2D tensor (channel, samples) for torchaudio.save
    if len(audio_sample.shape) == 1:
        audio_sample = audio_sample.unsqueeze(0)  # Add channel dimension
    
    # Save the waveform to a .wav file
    torchaudio.save(save_path, audio_sample, sampling_rate)
    print(f"Audio saved to {save_path}")

audio_sample = load_wav('audio/raw/2bhRZjDJ6k3UDWVVW36UGoV.wav', 24000)
meomeo_audio = enc_dec(audio_sample, meomeo, feature_extractor)
mimi_audio = enc_dec(audio_sample, mimi, feature_extractor)
save_wav(meomeo_audio, 'audio/out/meomeo_audio.wav', 24000)
save_wav(mimi_audio, 'audio/out/mimi_audio.wav', 24000)

