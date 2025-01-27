from tqdm import tqdm
import torch
import torchaudio
import scipy.signal
import numpy as np
import src.utils.logging as utils_logging

# this is a third version that joinly optimizes all speakers steering vector
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.FCP import FCP_torch_2, compute_inverse_power
from src.FCP import FCP_V1, FCP_V2
from src.IVA import IVA

class Sampler():

    def __init__(self, 
                 model, 
                 diff_params, 
                 args, 
                 xi=0, 
                 order=2,
                 n_fft=512, 
                 hop_length=128,
                 win_length=512,
                 lambda_reg = 0,
                 n_frames_past=0,
                 n_frames_future=0,
                 fcp_epsilon=1e-2,                 
                 n_spks=2,
                 n_ch=3,
                 
                 # use_warm_initialization=False, initialized_filter_step=0 -> not using IVA initialization
                 # use_warm_initialization=True, initialized_filter_step=0 -> only use source warm initialization
                 # use_warm_initialization=False, initialized_filter_step=200 -> only use IVA implied steering filter, need to tune number of steps
                 # use_warm_initialization=False, initialized_filter_step=total_steps -> only use IVA implied steering filter for all steps
                 # use_warm_initialization=True, initialized_filter_step=total_steps -> fully use initialization
                 use_warm_initialization=True,
                 warm_initialization_rescale=False,
                 warm_initialization_sigma=0.057,
                 initialized_filter_step=100, # set to zero if not using IVA implied FCP filter
                 
                 ref_loss_weight=0.5,
                 ref_loss_snr_threshold=0, # 3dB of for optimiation
                 ref_loss_max_step=100, # do not have ref channel loss when steps >= ref_loss_max_step
                 
                 ):

        # things needed for IVA intialization
        self.use_warm_initialization = use_warm_initialization
        self.warm_initialization_rescale = warm_initialization_rescale
        self.warm_initialization_sigma = warm_initialization_sigma
        self.initialized_filter_step = initialized_filter_step
        
        self.n_ch = n_ch
        self.n_spks = n_spks
        self.ref_loss_weight = ref_loss_weight
        self.ref_loss_snr_threshold = ref_loss_snr_threshold
        
        self.ref_loss_max_step = ref_loss_max_step
        
        self.fcp_epsilon = fcp_epsilon
        
        self.iva_in = min(self.n_ch, n_spks+1)

        self.IVA = IVA(
            n_fft = 2048,
            hop_length = 256,
            win_length = 2048,

            n_iter=100,
            n_src_iva = min(self.n_ch, n_spks+1),
            n_outs=n_spks,
            proj_back_mic=0,
            model='gauss', # guass or laplace or nmf, by default, it's laplace
            use_tiss=True,
            blank_frequency_start=1025
        )
        

        self.model = model
        self.diff_params = diff_params
        self.xi=xi #hyperparameter for the reconstruction guidance
        self.order=order
        self.args=args
        self.nb_steps=self.args.inference.T

        self.treshold_on_grads=args.inference.max_thresh_grads
        
        # add steering vector estimation and mixture resynthesis
        self.spatialization = FCP_V1(
            # n_fft=n_fft, 
            # hop_length=hop_length,
            # win_length=win_length,
            # lambda_reg = lambda_reg,
        
            n_fft=n_fft, 
            hop_length=hop_length,
            win_length=win_length,
            lambda_reg = lambda_reg,
            n_frames_past=n_frames_past,
            n_frames_future=n_frames_future,
            fcp_epsilon=self.fcp_epsilon
        
        )

    def get_score_rec_guidance(self, x, y, t_i, step, G_conj=None):
        # x: bs*n_spk, T
        # Y: bs, n_ch, T
        # bs, n_spk, n_ch, n_freq
        x.requires_grad_()
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        
        bs = y.shape[0]
        n_spk = int(x.shape[0] / bs)
        
        anchor_sources = x_hat.reshape(bs, n_spk, -1)
        anchor_mixture = anchor_sources.sum(1)

        _, all_channels_hat = self.spatialization(anchor_sources, y, G_conj)
        den_rec = all_channels_hat.sum(1)

        loss_all_channels = (den_rec - y).square().sum(-1).mean(1) # bs
        loss_ref_channel = (anchor_mixture - y[:, 0, :]).square().sum(-1) # bs
        
        snr = self.spatialization.calc_snr(y[:, 0, :], anchor_mixture).mean()
        if snr < self.ref_loss_snr_threshold and step < self.ref_loss_max_step:
            loss = loss_all_channels + self.ref_loss_weight * loss_ref_channel
        else:
            loss = loss_all_channels
        
        rec_grads=torch.autograd.grad(outputs=loss, inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/x.shape[-1]**0.5
        
        #normalize scaling
        s=self.xi/(normguide*t_i+1e-6)
        
        #optionally apply a treshold to the gradients
        if self.treshold_on_grads>0:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        

        score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        score=score-s*rec_grads
        # score=-s*rec_grads

        return score

    def get_score(self,x, y, t_i, step, G_conj=None):
        if y==None:
            #unconditional sampling
            with torch.no_grad():
                # print(t_i.unsqueeze(-1).shape)
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                score=(x_hat-x)/t_i**2
            return score
        else:
            # if self.xi>0:
                #apply rec. guidance
            score=self.get_score_rec_guidance(x, y, t_i, step, G_conj)
    
            return score

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device
    ):
        self.y=None
        return self.predict(shape, device)
    
    def iva_initialize(self, y, sigma):
        # y: bs, n_ch, T
        # iva_outs = self.IVA(y[:, :self.iva_in, :]) # bs, n_spk, # reference channel outputs (high STFT bins are blanked)
        iva_outs = self.IVA(y) # bs, n_spk, # reference channel outputs (high STFT bins are blanked)
        if not self.use_warm_initialization:
            return torch.randn(iva_outs.shape).to(y.device)*sigma, iva_outs
        if self.warm_initialization_rescale:
            x = self.warm_initialization_sigma * iva_outs / iva_outs.std(-1, keepdim=True)
            x = x + torch.randn(x.shape).to(y.device)*sigma
        else:
            x = iva_outs
            x = x + torch.randn(x.shape).to(y.device)*sigma
        
        return x, iva_outs
    
    def separate(
        self,
        y, # bs, n_ch, T
        n_spk,
        device,
        # w_steer_filter=None
    ):

        self.y = y
        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        bs, n_ch, T = y.shape
        shape = (bs*n_spk, T)
        
        # use iva to warm initialize the diffusion sampling
        x, iva_out = self.iva_initialize(y, t[0]) # noisified iva outputs, low frequency iva + noise, high frequency pure noise (bs, n_spk, T)
        x = x.reshape(bs*n_spk, -1)
        # x = self.diff_params.sample_prior(shape,t[0]).to(device) # bs, n_spk, T
        
        # calcualte the G_conj_sub, the initialized FCP filter will be used later on
        G_conj, _ = self.spatialization(iva_out, y) # initialized FCP filter

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)


        for i in tqdm(range(0, self.nb_steps, 1)):
            #print("sampling step ",i," from ",self.nb_steps)

            if gamma[i]==0:
                #deterministic sampling, do nothing
                t_hat=t[i] 
                x_hat=x
            else:
                #stochastic sampling
                #move timestep
                t_hat=t[i]+gamma[i]*t[i] 
                #sample noise, Snoise is 1 by default
                epsilon=torch.randn(shape).to(device)*self.diff_params.Snoise
                #add extra noise
                x_hat=x+((t_hat**2 - t[i]**2)**(1/2))*epsilon 


            if i >= self.initialized_filter_step:
                score=self.get_score(x_hat, self.y, t_hat, i)
            else:
                score=self.get_score(x_hat, self.y, t_hat, i, G_conj)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                if i >= self.initialized_filter_step:
                    score=self.get_score(x_prime, self.y, t_prime, i)
                else:
                    score=self.get_score(x_prime, self.y, t_prime, i, G_conj)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d
            
        return x.detach().view(bs, n_spk, -1)

    def predict(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device, #lambda function
    ):

        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape,t[0]).to(device)

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)


        for i in tqdm(range(0, self.nb_steps, 1)):
            # print("sampling step ",i," from ",self.nb_steps)

            if gamma[i]==0:
                #deterministic sampling, do nothing
                t_hat=t[i] 
                x_hat=x
            else:
                #stochastic sampling
                #move timestep
                t_hat=t[i]+gamma[i]*t[i] 
                #sample noise, Snoise is 1 by default
                epsilon=torch.randn(shape).to(device)*self.diff_params.Snoise
                #add extra noise
                x_hat=x+((t_hat**2 - t[i]**2)**(1/2))*epsilon 

            score=self.get_score(x_hat, self.y, t_hat)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, self.y, t_prime)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d
            
        return x.detach()