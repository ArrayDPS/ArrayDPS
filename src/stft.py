import torch

class STFT:
    def __init__(self, 
                 n_fft=512, 
                 hop_length=100,
                 win_length=400,
                 ):
        super(STFT, self).__init__()
        self.windows = {}
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def STFT(self, input):
        # B, T
        device = input.device
        if device not in self.windows.keys():
            self.windows[device] = torch.hann_window(self.win_length).sqrt().to(device)
        return torch.stft(
            input, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length,
            window=self.windows[device], 
            center=True, 
            pad_mode='constant', 
            return_complex=True
        )
        
    def ISTFT(self, input):
        device = input.device
        if device not in self.windows.keys():
            self.windows[device] = torch.hann_window(self.win_length).sqrt().to(device)
        return torch.istft(
            input, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length,
            window=self.windows[device]
            )