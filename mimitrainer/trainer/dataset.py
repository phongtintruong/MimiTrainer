from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import torchaudio.transforms as T
import os



# def collate_fn(data):
#     # return pad_sequence(data, batch_first=True)
#     # return pad_sequence(*data)
#     is_one_data = not isinstance(data[0], tuple)
#     outputs = []
#     if is_one_data:
#         for datum in data:
#             if isinstance(datum, torch.Tensor):
#                 output = datum.unsqueeze(0)
#             else:
#                 output = torch.tensor([datum])
#             outputs.append(output)
#         return tuple(outputs)
#     for datum in zip(*data):
#         if isinstance(datum[0], torch.Tensor):
#             output = pad_sequence(datum, batch_first=True)
#         else:
#             output = torch.tensor(list(datum))
#         outputs.append(output)
#
#     return tuple(outputs)
#
#
# def get_dataloader(ds, **kwargs):
#     return DataLoader(ds, collate_fn=collate_fn, **kwargs)


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



class DistillDataset(Dataset):
    def __init__(self, data_dir, sample_rate_student, student_feature_extractor, sample_rate_teacher, teacher_feature_extractor, mode='train'):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate_student = sample_rate_student
        self.student_feature_extractor = student_feature_extractor
        self.sample_rate_teacher = sample_rate_teacher
        self.teacher_feature_extractor = teacher_feature_extractor
        self.mode = mode

        # Find all audio files in the data_dir
        self.audio_files = self.find_audio_files(data_dir)

    def find_audio_files(self, data_dir):
        """Finds all audio files of supported extensions in a directory recursively."""
        audio_files = []
        supported_extensions = (".wav", ".flac", ".mp3", ".ogg")  # Add other extensions if needed
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]  # Directly use the path from self.audio_files
        waveform, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate_student:
            waveform_student = T.Resample(orig_freq=sr, new_freq=self.sample_rate_student)(waveform)
        else:
            waveform_student = waveform

        if sr != self.sample_rate_teacher:
            waveform_teacher = T.Resample(orig_freq=sr, new_freq=self.sample_rate_teacher)(waveform)
        else:
            waveform_teacher = waveform

        # If the audio has multiple channels, take the mean to make it mono
        if waveform_student.shape[0] > 1:
            waveform_student = waveform_student.mean(dim=0, keepdim=True)
        if waveform_teacher.shape[0] > 1:
            waveform_teacher = waveform_teacher.mean(dim=0, keepdim=True)

        inputs_student = self.student_feature_extractor(raw_audio=waveform_student.squeeze(0).numpy(), sampling_rate=self.sample_rate_student, return_tensors='pt')["input_values"]
        inputs_teacher = self.teacher_feature_extractor(waveform_teacher.squeeze(0).numpy(), sampling_rate=self.sample_rate_teacher, return_tensors='pt')["input_values"]

        # print(f"Item {idx}: inputs_student.shape = {inputs_student.shape}, inputs_teacher.shape = {inputs_teacher.shape}")

        return waveform_student, inputs_student, inputs_teacher


def collate_fn(data):
    """
    Collate function to pad waveforms, student inputs, and teacher inputs.

    Args:
        data (list): List of (waveform_student, inputs_student, inputs_teacher) tuples.

    Returns:
        tuple: Padded student waveforms, student inputs, and teacher inputs.
    """
    waveforms, student_inputs, teacher_inputs = zip(*data)

    # Pad waveforms along the time dimension
    waveforms_padded = pad_sequence(
        [waveform.squeeze(0) for waveform in waveforms],  # [1, seq_len] -> [seq_len]
        batch_first=True,
        padding_value=0.0
    ).unsqueeze(1)

    # Pad teacher inputs along the time dimension
    teacher_inputs_padded = pad_sequence(
        [x.squeeze(0) for x in teacher_inputs],  # [1, seq_len] -> [seq_len]
        batch_first=True,
        padding_value=0.0
    )  # -> [batch_size, max_seq_len_teacher]

    # Pad student inputs along the time dimension
    student_inputs_padded = pad_sequence(
        [x.squeeze(0).squeeze(0) for x in student_inputs],  # [1, 1, seq_len] -> [seq_len]
        batch_first=True,
        padding_value=0.0
    ).unsqueeze(1)  # Add back channel dimension -> [batch_size, 1, max_seq_len_student]

    return waveforms_padded, student_inputs_padded, teacher_inputs_padded


def get_dataloader(dataset, batch_size, num_workers=4, shuffle=True, drop_last=True):  # Add drop_last parameter
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)