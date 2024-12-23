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
    def __init__(self, data_dir, mode='train'):
        super().__init__()
        self.data_dir = data_dir
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

        return waveform, sr


def collate_fn(batch, feature_extractor_teacher, feature_extractor_student, teacher_sampling_rate, student_sampling_rate, max_length_s=3):
    """
    Args:
        batch (list): List of raw audio waveforms from Dataset.
        feature_extractor_teacher (callable): Feature extractor for the teacher model.
        feature_extractor_student (callable): Feature extractor for the student model.
        teacher_sampling_rate (int): Sampling rate for the teacher.
        student_sampling_rate (int): Sampling rate for the student.
        max_length_s (int): Maximum audio length in seconds.

    Returns:
        features_teacher (torch.Tensor): Preprocessed features for teacher.
        features_student (torch.Tensor): Preprocessed features for student.
    """
    teacher_processed_batch = []
    student_processed_batch = []
    for waveform, sr in batch:
        if sr != student_sampling_rate:
            waveform_student = T.Resample(orig_freq=sr, new_freq=student_sampling_rate)(waveform)
        else:
            waveform_student = waveform

        if sr != teacher_sampling_rate:
            waveform_teacher = T.Resample(orig_freq=sr, new_freq=teacher_sampling_rate)(waveform)
        else:
            waveform_teacher = waveform

        # Ensure waveform is mono
        if waveform_student.shape[0] > 1:
            waveform_student = waveform_student.mean(dim=0)
        else:
            waveform_student = waveform_student.squeeze(0)

        if waveform_teacher.shape[0] > 1:
            waveform_teacher = waveform_teacher.mean(dim=0)
        else:
            waveform_teacher = waveform_teacher.squeeze(0)

        teacher_processed_batch.append(waveform_teacher.numpy())
        student_processed_batch.append(waveform_student.numpy())
        print('waveform_teacher shape')
        # print(waveform.shape)
        print(waveform_teacher.shape)
    #
    # Extract features for both teacher and student
    features_teacher = feature_extractor_teacher(
        teacher_processed_batch,
        sampling_rate=teacher_sampling_rate,
        max_length=teacher_sampling_rate * max_length_s,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )["input_values"]

    # print(features_student.shape)
    # print(features_teacher.shape)
    # Compute max_length in terms of sample count
    max_length_samples = int(student_sampling_rate * max_length_s)

    # Check if all samples are shorter than max_length
    all_shorter_than_max_length = all(len(waveform) <= max_length_samples for waveform in student_processed_batch)

    if all_shorter_than_max_length:
        # Use padding when all samples are short
        features_student = feature_extractor_student(
            student_processed_batch,
            sampling_rate=student_sampling_rate,
            max_length=max_length_samples,
            padding=True,  # Pad to the same length
            return_tensors="pt"
        )["input_values"]
    else:
        # Use truncation when some samples exceed max_length
        features_student = feature_extractor_student(
            student_processed_batch,
            sampling_rate=student_sampling_rate,
            max_length=max_length_samples,
            truncation=True,  # Truncate longer samples
            return_tensors="pt"
        )["input_values"]

    return features_student, features_teacher


def get_dataloader(dataset, batch_size, feature_extractor_teacher, feature_extractor_student, teacher_sampling_rate, student_sampling_rate, max_length_s=3, num_workers=4, shuffle=True, drop_last=True):  # Add drop_last parameter
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=lambda batch: collate_fn(batch, feature_extractor_teacher, feature_extractor_student, teacher_sampling_rate, student_sampling_rate, max_length_s))