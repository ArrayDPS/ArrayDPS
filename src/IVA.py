import torch.nn as nn
import torch.nn.functional as F

import torch
import torchiva
from sdr import batch_SDR_torch
from src.stft import STFT

'''
for reverberation speaker separation, recommending:
    n_fft = 2048,
    hop_length = 256,
    win_length = 2048,

    n_iter=100,
    n_src_iva = 3,
    n_outs=2,
    proj_back_mic=0,
    model='gauss', # guass or laplace or nmf, by default, it's laplace
    use_tiss=True,
'''
class IVA(nn.Module):
    def __init__(self, 
                n_fft = 2048,
                hop_length = 256,
                win_length = 2048,

                n_iter=100,
                n_src_iva = 3,
                n_outs=2,
                proj_back_mic=0,
                model='gauss', # guass or laplace or nmf, by default, it's laplace
                use_tiss=False,
                
                blank_frequency_start=1025
                ):
        super(IVA, self).__init__()
        
        self.blank_frequency_start = blank_frequency_start
        
        self.n_src_iva = n_src_iva
        self.n_outs = n_outs
        
        if model == 'gauss':
            model_prior = torchiva.models.GaussModel()
        elif model == 'nmf':
            model_prior = torchiva.models.NMFModel()
        else:
            model_prior = None

        self.stft = STFT(
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
        )

        if use_tiss:
            self.separator = torchiva.T_ISS(
            n_iter=n_iter,
            n_taps=0,
            n_delay=0,
            n_src=n_src_iva,
            model=model_prior,
            proj_back_mic=proj_back_mic,
            use_dmc=False,
            eps=None,
            )#.to(device)
        else:
            self.separator = torchiva.AuxIVA_IP(
                n_iter=n_iter,
                n_src=n_src_iva,
                proj_back_mic=proj_back_mic,
                model=model_prior
                # eps=1e-5
            )
        
    def forward(self, mixture):
        
        # mixture: bs, n_ch, T
        
        
        X = F.pad(mixture, (0, self.stft.n_fft)) # bs, n_ch, T
        bs, n_ch, T = X.shape
        X = X.view(bs*n_ch, T)
        X_STFT = self.stft.STFT(X) # bs*n_ch, n_freq, n_frame
        _, n_freq, n_frame = X_STFT.shape
        X_STFT = X_STFT.view(bs, n_ch, n_freq, n_frame) # bs, n_ch, n_freq, n_frame
        
        X_STFT = X_STFT[:, :, :self.blank_frequency_start, :] # bs, n_ch, n_freq_low, n_frame
        
        Y_STFT = self.separator(X_STFT) # bs, n_src_iva, n_freq_low, n_frame
        
        Y_STFT = torch.cat([Y_STFT, torch.zeros(bs, self.n_src_iva, n_freq-self.blank_frequency_start, n_frame).to(mixture.device)], dim=2) # bs, n_src_iva, n_freq, n_frame
        
        Y = self.stft.ISTFT(Y_STFT.view(-1, n_freq, n_frame)) #  bs* n_src_iva, T
        Y = Y.view(bs,self.n_src_iva, -1)[..., :mixture.shape[-1]] # bs, n_src_iva, T
        
        if self.n_src_iva > self.n_outs:
            energy = torch.sum(Y**2, dim=2) # bs, n_src_iva
            top_indices = torch.topk(energy, self.n_outs, dim=1).indices  # Shape: (bs, n_src_iva)
            batch_indices = torch.arange(energy.size(0)).unsqueeze(-1)  # Shape: (batch_size, 1)
            Y = Y[batch_indices, top_indices] # bs, n_outs
        
        return Y