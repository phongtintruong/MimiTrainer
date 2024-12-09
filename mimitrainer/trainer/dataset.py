from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np


def collate_fn(data):
    # return pad_sequence(data, batch_first=True)
    # return pad_sequence(*data)
    is_one_data = not isinstance(data[0], tuple)
    outputs = []
    if is_one_data:
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            output = pad_sequence(datum, batch_first=True)
        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)


def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)


class audioDataset(Dataset):

    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 valid=False):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.valid = valid

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split('\t')  # Assuming tab-separated file paths
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file))

        # If the audio has multiple channels, take the mean to make it mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample the audio if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        # Handle audio and feature segment extraction
        if audio.shape[1] > self.segment_size:
            if self.valid:
                # For validation, take the first segment
                audio = audio[:, :self.segment_size]
                feature = feature[:self.segment_size, :]
            else:
                # For training, take a random segment of the audio
                max_audio_start = audio.shape[1] - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start + self.segment_size]

                # Adjust feature extraction to match the audio segment
                feature_start = audio_start
                feature_end = min(feature_start + self.segment_size, feature.shape[0])
                feature = feature[feature_start:feature_end, :]
        else:
            # If audio is shorter than the segment size, pad it if training
            if not self.valid:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[1]), 'constant')
            # If feature is shorter than audio, pad it
            if feature.shape[0] < self.segment_size:
                feature = torch.nn.functional.pad(feature, (0, 0, 0, self.segment_size - feature.shape[0]), 'constant')

        return audio, feature