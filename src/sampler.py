from tqdm import tqdm
import torch
import torchaudio
import scipy.signal
import numpy as np
import src.utils.logging as utils_logging
class Sampler():

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):

        self.model = model
        self.diff_params = diff_params
        self.xi=xi #hyperparameter for the reconstruction guidance
        self.order=order
        self.data_consistency=data_consistency #use reconstruction gudance without replacement
        self.args=args
        self.nb_steps=self.args.inference.T

        self.treshold_on_grads=args.inference.max_thresh_grads
        self.rid=rid


    def data_consistency_step_phase_retrieval(self, x_hat, y):
        """
        Data consistency step only for the phase retrieval case, we are replacing the observed magnitude
        """
        #apply replacment (valid for linear degradations)
        win_size=self.args.inference.phase_retrieval.win_size
        hop_size=self.args.inference.phase_retrieval.hop_size

        window=torch.hamming_window(window_length=win_size).to(x_hat.device)
        #print(x.shape)
        x2=torch.cat((x_hat, torch.zeros(x_hat.shape[0],win_size ).to(x_hat.device)),-1)
        X=torch.stft(x2, win_size, hop_length=hop_size,window=window,center=False,return_complex=True)
        phaseX=torch.angle(X)
        assert y.shape == phaseX.shape, y.shape+" "+phaseX.shape
        X_out=torch.polar(y, phaseX)
        X_out=torch.view_as_real(X_out)
        x_out=torch.istft(X_out, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        #print(x_hat.shape)
        x_out=x_out[...,0:x_hat.shape[-1]]
        #print(x_hat.shape, x.shape)
        assert x_out.shape == x_hat.shape
        return x_out

    def data_consistency_step(self, x_hat, y, degradation):
        """
        Simple replacement method, used for inpainting and FIR bwe
        """
        #get reconstruction estimate
        den_rec= degradation(x_hat)     
        #apply replacment (valid for linear degradations)
        return y+x_hat-den_rec 

    def get_score_rec_guidance(self, x, y, t_i, degradation):

        x.requires_grad_()
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        den_rec= degradation(x_hat) 

        if len(y.shape)==3:
            dim=(1,2)
        elif len(y.shape)==2:
            dim=1

        norm=torch.linalg.norm(y-den_rec,dim=dim, ord=2)
        
        rec_grads=torch.autograd.grad(outputs=norm,
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/self.args.audio_len**0.5
        
        #normalize scaling
        s=self.xi/(normguide*t_i+1e-6)
        
        #optionally apply a treshold to the gradients
        if self.treshold_on_grads>0:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        

        score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        score=score-s*rec_grads

        return score

    def get_score(self,x, y, t_i, degradation):
        if y==None:
            assert degradation==None
            #unconditional sampling
            with torch.no_grad():
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i, degradation)
    
                #optionally apply replacement or consistency step
                if self.data_consistency:
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
                    if self.args.inference.mode=="phase_retrieval":
                        x_hat=self.data_consistency_step_phase_retrieval(x_hat,y)
                    else:
                        x_hat=self.data_consistency_step(x_hat,y, degradation)
    
                    #convert back to score
                    score=(x_hat-x)/t_i**2
    
            else:
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                        
                    x_hat=self.data_consistency_step(x_hat,y, degradation)
        
                    score=(x_hat-x)/t_i**2
    
            return score

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device
    ):
        self.y=None
        self.degradation=None
        return self.predict(shape, device)

    def predict_resample(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        shape,
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        print(shape)
        return self.predict(shape, y.device)


    def predict_conditional(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        return self.predict(y.shape, y.device)

    def predict(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device, #lambda function
    ):

        if self.rid:
            data_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))

        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape,t[0]).to(device)

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

            score=self.get_score(x_hat, self.y, t_hat, self.degradation)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, self.y, t_prime, self.degradation)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d

            if self.rid: data_denoised[i]=x
            
        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()