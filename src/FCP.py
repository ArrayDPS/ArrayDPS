import torch
from torch.nn import functional as F


def build_y_tilde_t(x, taps, delay):
    # x.shape [n_batch, n_freqs, n_mics, n_frames]
    assert delay == 0
    assert type(taps) is int
    n_mics, n_frames = x.shape[-2:]
    other_dims = list(x.shape[:-2])
    x = x.view([-1,n_mics,n_frames,1])

    x = F.pad(x, (0,0,taps-1,0)) # [-1,n_mics,taps-1+n_frames,1]
    x = F.unfold(x, (taps,1)) # [-1, n_mics*taps, n_frames]

    x = x.view([-1,n_mics,taps,n_frames])
    x = x.transpose(1,2).contiguous() # [-1,taps,n_mics,n_frames]
    x = x.view(other_dims+[taps*n_mics,n_frames])
    return x


def FCP_torch_2(mix_cs, sph_cs, frame_range_,
        taps=20, delay=0, inverse_power=None,
        weight_type=0,
        diagload=1e-5,
        ):
    """FCP"""
    #
    # mix_cs.shape is  [n_batch, n_frames, n_mics, n_freqs]
    # sph_cs.shape is  [n_batch, n_frames, n_srcs, n_freqs]
    # inverse_power is [n_batch, n_frames,      1, n_freqs]
    #
    n_batch, n_frames, n_mics, n_freqs = mix_cs.shape
    n_srcs = sph_cs.shape[-2]
    #
    mix_cs = mix_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_mics, n_frames]
    sph_cs = sph_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_srcs, n_frames]
    inverse_power = inverse_power.transpose(1,3) #  [n_batch, n_freqs,      1, n_frames]

    Y = mix_cs # [n_batch, n_freqs, n_mics, n_frames]
    if delay == 0:
        Y_tilde = build_y_tilde_t(sph_cs, taps, delay) # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay < 0:
        tmp_sph_cs = F.pad(sph_cs, (0,-delay)) # [n_batch, n_freqs, n_srcs, n_frames+(-delay)]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, n_frames+(-delay)]
        Y_tilde = Y_tilde[...,-delay:] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay > 0:
        tmp_sph_cs = F.pad(sph_cs, (delay,0)) # [n_batch, n_freqs, n_srcs, delay+n_frames]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, delay+n_frames]
        Y_tilde = Y_tilde[...,:-delay] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    else:
        raise

    Y_tilde = Y_tilde[...,frame_range_] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    Y = Y[...,frame_range_] # [n_batch, n_freqs, n_mics, n_frames]
    n_frames = Y_tilde.shape[-1]

    Y_tilde = Y_tilde.view([n_batch, n_freqs, taps, n_srcs, n_frames])
    Y_tilde = Y_tilde.permute(0,1,3,4,2) # [n_batch, n_freqs, n_srcs, n_frames, taps]

    inverse_power = inverse_power[:,:,None]                    # [n_batch, n_freqs,      1,    1, n_frames]
    weighted_Y_tilde = Y_tilde.swapaxes(-1,-2) * inverse_power # [n_batch, n_freqs, n_srcs, taps, n_frames]

    R = torch.matmul(weighted_Y_tilde, Y_tilde.conj()) / n_frames # [n_batch, n_freqs, n_srcs, taps, taps]

    Y = Y.swapaxes(-1, -2) # [n_batch, n_freqs, n_frames, n_mics]
    Y = Y[:,:,None] # [n_batch, n_freqs, 1, n_frames, n_mics]

    P = torch.matmul(weighted_Y_tilde, Y.conj()) / n_frames # [n_batch, n_freqs, n_srcs, taps, n_mics]

    old_dtype = R.dtype
    R = R.to(torch.complex128)
    P = P.to(torch.complex128)
    G = torch.linalg.solve(R+diagload*torch.eye(R.shape[-1],dtype=R.dtype,device=R.device), P) # [n_batch, n_freqs, n_srcs, taps, n_mics]
    G = G.to(old_dtype) 

    ret = torch.matmul(Y_tilde, G.conj()) # [n_batch, n_freqs, n_srcs, n_frames, n_mics]
    ret = ret.permute(0,3,2,4,1) # [n_batch, n_frames, n_srcs, n_mics, n_freqs]
    return ret, G.conj()

# this function directly filters using given G_conj
def FCP_filter(sph_cs, G_conj, taps=20, delay=0):
    # ph_cs.shape is  [n_batch, n_frames, n_srcs, n_freqs]
    # G_conj.shape is [n_batch, n_freqs, n_srcs, taps, n_mics]

    n_batch, n_frames, _, n_freqs = sph_cs.shape
    n_srcs = sph_cs.shape[-2]

    sph_cs = sph_cs.permute(0,3,2,1).contiguous() # [n_batch, n_freqs, n_srcs, n_frames]

    if delay == 0:
        Y_tilde = build_y_tilde_t(sph_cs, taps, delay) # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay < 0:
        tmp_sph_cs = F.pad(sph_cs, (0,-delay)) # [n_batch, n_freqs, n_srcs, n_frames+(-delay)]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, n_frames+(-delay)]
        Y_tilde = Y_tilde[...,-delay:] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    elif delay > 0:
        tmp_sph_cs = F.pad(sph_cs, (delay,0)) # [n_batch, n_freqs, n_srcs, delay+n_frames]
        Y_tilde = build_y_tilde_t(tmp_sph_cs, taps, 0) # [..., taps*n_srcs, delay+n_frames]
        Y_tilde = Y_tilde[...,:-delay] # [n_batch, n_freqs, taps*n_srcs, n_frames]
    else:
        raise

    n_frames = Y_tilde.shape[-1]

    Y_tilde = Y_tilde.view([n_batch, n_freqs, taps, n_srcs, n_frames])
    Y_tilde = Y_tilde.permute(0,1,3,4,2) # [n_batch, n_freqs, n_srcs, n_frames, taps]

    ret = torch.matmul(Y_tilde, G_conj) # [n_batch, n_freqs, n_srcs, n_frames, n_mics]
    ret = ret.permute(0,3,2,4,1) # [n_batch, n_frames, n_srcs, n_mics, n_freqs]
    return ret

def compute_inverse_power(Y, max_=None, fcp_weight_type=0, fcp_epsilon=1e-3):
    # Y.shape is [n_batch, n_frames, n_imics, n_freqs]
    power = torch.mean(Y.real**2+Y.imag**2,dim=-2,keepdim=True) # [n_batch, n_frames, 1, n_freqs]
    if max_ is None:
        if fcp_weight_type in [0,1,]:
            max_, _ = torch.max(power,dim=1,keepdim=True) # [n_batch, 1, 1, n_freqs]
            max_, _ = torch.max(max_,dim=3,keepdim=True) # [n_batch, 1, 1, 1]
        else:
            raise
    if fcp_weight_type in [0,]:
        return 1. / (power + max_*fcp_epsilon + 1e-6), max_ # [n_batch, n_frames, 1, n_freqs]
    elif fcp_weight_type in [1,]:
        return 1. / torch.maximum(power + 1e-6, max_*fcp_epsilon), max_ # [n_batch, n_frames, 1, n_freqs]
    else:
        raise

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCP_V1(nn.Module):
    def __init__(self, 
                 n_fft=512, 
                 hop_length=128,
                 win_length=512,
                 lambda_reg = 2e-4,
                 n_frames_past=20,
                 n_frames_future=0,
                 fcp_epsilon=1e-3
                 ):
        super(FCP_V1, self).__init__()
        self.windows = {}
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.lambda_reg = lambda_reg
        
        self.fcp_epsilon = fcp_epsilon
        
        self.n_frames_past = n_frames_past
        self.n_frames_future = n_frames_future

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
        
    def calc_snr(self, s, s_hat, epsilon=1e-12):
        error_energy = (s - s_hat).square().sum(-1)
        signal_enery = s.square().sum(-1)
        return 10*(torch.log10(signal_enery+epsilon) - torch.log10(error_energy+epsilon))
    
    def forward(self, ref_channel, mixture, G_conj=None):
        # sources: bs, n_spk, T
        # mixture: bs, n_ch, T
        bs, n_spk, len_orig = ref_channel.shape
        n_ch = mixture.shape[1]
        
        ref_channel = F.pad(ref_channel, (0, self.n_fft))
        mixture = F.pad(mixture, (0, self.n_fft))
        
        ref_channel_spec = self.STFT(
            ref_channel.view(bs*n_spk, -1)
        ) # bs*n_spk, n_freq, n_frame
            
        mixture_spec = self.STFT(
            mixture.view(bs*n_ch, -1)
        ) # bs*n_ch, n_freq, n_frame
        
        n_freq, n_frame = ref_channel_spec.shape[-2], ref_channel_spec.shape[-1]

        ref_channel_spec = ref_channel_spec.view(bs, n_spk, n_freq, n_frame).permute(0,3,1,2) # bs, n_frame, n_spk, n_freq
        
        mixture_spec = mixture_spec.view(bs, n_ch, n_freq, n_frame).permute(0,3,1,2) # bs, n_frame, n_ch, n_freq

        if G_conj is not None:
            # G_conj [n_batch, n_freqs, n_srcs, taps, n_mics]
            # mixture_spec_nb = mixture_spec[..., :nb]            
            all_channels_spec_hat = FCP_filter(ref_channel_spec, G_conj, taps=self.n_frames_past, delay=self.n_frames_future)
        else:
            inverse_power, _ = compute_inverse_power(mixture_spec, None, 0, self.fcp_epsilon) # [n_batch, n_frames, 1, n_freqs], [n_batch, 1, 1, 1]
            # inverse_power = torch.ones(inverse_power.shape)
            frame_range_ = slice(0, n_frame, None)

            all_channels_spec_hat, G_conj = FCP_torch_2(
                    mixture_spec, # [n_batch, n_frames, n_imics, n_freqs]
                    ref_channel_spec, # [n_batch, n_frames, n_srcs, n_freqs]
                    frame_range_,
                    taps=self.n_frames_past, delay=self.n_frames_future, inverse_power=inverse_power,
                    weight_type=0,
                    diagload=self.lambda_reg
                    )       # [n_batch, n_frames, n_srcs, n_imics, n_freqs]
        
        all_channels_spec_hat = all_channels_spec_hat.permute(0,2,3,4,1)
        
        all_channels_spec_hat = all_channels_spec_hat.reshape(bs*n_spk*n_ch, n_freq, n_frame)

        all_channels_hat = self.ISTFT(all_channels_spec_hat).view(bs, n_spk, n_ch, -1)

        all_channels_hat = all_channels_hat[..., :len_orig]
        
        return G_conj, all_channels_hat
