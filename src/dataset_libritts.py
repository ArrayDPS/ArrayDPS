import torch
from torch.utils.data import Dataset, ConcatDataset
import torchaudio
import random

from scipy.io import wavfile
import numpy as np
from scipy import signal
import os
import pandas as pd
from torch.utils.data import IterableDataset
import numpy as np

class LIBRITTS_TrainSet(Dataset):
    def __init__(self, 
                 root, 
                 urls=None, 
                 download=False, 
                 audio_len=16000*3, 
                 min_audio_len=16000*1,
                 target_sampling_rate=None, 
                 std_norm=True, 
                 std=0.057):
        """
        Initialize the dataset by concatenating multiple subsets with defaults to all available training data,
        and allow resampling to a target sampling rate.
        
        Args:
            root (str): Path to the directory where the dataset is stored or will be downloaded.
            urls (list of str): List of subsets of the dataset to use. Defaults to all main subsets.
            download (bool): Whether to download the dataset if it's not already available locally.
            subset (str): 'train' for training set, 'val' or 'test' for validation or testing set.
            target_duration (float): Target duration for each audio sample in seconds.
            target_sampling_rate (int): Target sampling rate to which all audio files will be resampled.
        """
        if urls is None:
            urls = [
                'train-clean-100'#, 'train-clean-360'
            ]
        
        self.datasets = [torchaudio.datasets.LIBRITTS(root=root, url=url, download=download) for url in urls]
        self.dataset = ConcatDataset(self.datasets)
        self.audio_len = audio_len
        self.min_audio_len = min_audio_len
        self.target_sampling_rate = target_sampling_rate
        
        self.resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=self.target_sampling_rate)
        
        self.std_norm = std_norm
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, utterance, _, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        # Resample to the target sampling rate if specified
        if self.target_sampling_rate is not None and sample_rate != self.target_sampling_rate:
            waveform = self.resampler(waveform)
            # origin_len = waveform.shape[-1]
            if waveform.shape[-1] < self.min_audio_len:
                return self.__getitem__(np.random.randint(0, self.__len__()))
            sample_rate = self.target_sampling_rate  # Update the sample rate to the target rate

        # Calculate number of samples needed for the target duration
        num_target_samples = self.audio_len

        # Pad or truncate the waveform to the target number of samples
        if waveform.shape[1] < num_target_samples:
            # Zero-padding
            padding_size = num_target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))
        elif waveform.shape[1] > num_target_samples:
            # Truncation: Randomly sample a starting point
            start_sample = random.randint(0, waveform.shape[1] - num_target_samples)
            waveform = waveform[:, start_sample:start_sample + num_target_samples]
            
        # waveform = waveform / (waveform.max() + 1e-8)
        # max_val = random.uniform(self.max_range[0], self.max_range[1])
        
        # if self.max_norm:
        #     waveform = waveform * max_val
        
        if self.std_norm:
            waveform = waveform / (waveform.std() + 1e-8)
            waveform = self.std * waveform

        return waveform[0]#, origin_len

class LIBRITTS_IterableDataset(IterableDataset):
    def __init__(self, dataset: LIBRITTS_TrainSet):
        """
        Args:
            dataset (LIBRITTS_TrainSet): The underlying dataset from which to sample.
        """
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        while True:
            # Randomly select an index from the dataset
            idx = random.randint(0, len(self.dataset) - 1)
            # Fetch the sample corresponding to the random index
            yield self.dataset[idx]