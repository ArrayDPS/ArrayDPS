import os
import torch
import torchaudio

class SMSWSJDataset(torch.utils.data.Dataset):
    def __init__(self, smswsj_dir, sample_rate=8000, num_channels=3):
        """
        Args:
            smswsj_dir (str): Path to the dataset directory containing 'early', 'tail', and 'observation' folders.
            sample_rate (int): Sampling rate of the audio files (default: 8000).
            num_channels (int): Number of channels to load (3 or 6). If 3, use indices 0, 2, 4.
        """
        self.smswsj_dir = smswsj_dir
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.indices = [0, 2, 4] if num_channels == 3 else list(range(num_channels))
        self.test_eval92_dir = "test_eval92"

        # Get the list of files from the observation folder
        self.observation_files = sorted(
            os.listdir(os.path.join(smswsj_dir, "observation", self.test_eval92_dir))
        )

    def __len__(self):
        return len(self.observation_files)

    def __getitem__(self, idx):
        # Get filenames for observation, early, and tail
        obs_file = self.observation_files[idx]
        base_name = os.path.splitext(obs_file)[0]  # Remove file extension

        # Construct file paths
        obs_path = os.path.join(self.smswsj_dir, "observation", self.test_eval92_dir, obs_file)
        early_paths = [
            os.path.join(self.smswsj_dir, "early", self.test_eval92_dir, f"{base_name}_{i}.wav")
            for i in range(2)
        ]
        tail_paths = [
            os.path.join(self.smswsj_dir, "tail", self.test_eval92_dir, f"{base_name}_{i}.wav")
            for i in range(2)
        ]

        # Load audio
        observation, _ = torchaudio.load(obs_path)
        early = [torchaudio.load(path)[0] for path in early_paths]
        tail = [torchaudio.load(path)[0] for path in tail_paths]

        # Select required channels
        observation = observation[self.indices, :]
        early = torch.stack([audio[self.indices, :] for audio in early], dim=0)
        tail = torch.stack([audio[self.indices, :] for audio in tail], dim=0)

        return observation, early, tail
