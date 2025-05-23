import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math as m
import torch
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

import torchaudio

from .ncsnpp_utils import  layerspp
class CombinerUp(nn.Module):

    def __init__(self,mode, Npyr, Nx, bias=True):
        super().__init__()
        self.conv1x1=nn.Conv2d(Nx, Npyr,1, bias=bias)
        self.mode=mode
        #self.GN=nn.GroupNorm(8,Nx)
        torch.nn.init.constant_(self.conv1x1.weight, 0)
    def forward(self,pyr,x):
                
        if self.mode=="sum":
            x=self.conv1x1(x)
            if pyr==None:
                return x
            else:
                return (pyr+x)/(2**0.5)
        else:
            raise NotImplementedError

class CombinerDown(nn.Module):

    def __init__(self,mode, Nin, Nout, bias=True):
        super().__init__()
        self.conv1x1=nn.Conv2d(Nin, Nout,1, bias=bias)
        self.mode=mode
    def forward(self,pyr,x):
        if self.mode=="sum":
            pyr=self.conv1x1(pyr)
            return (pyr+x)/(2**0.5)
        else:
            raise NotImplementedError

class Upsample(nn.Module):
    def __init__(self,S):
        super().__init__()
        self.resample = layerspp.Upsample( with_conv=False, fir=True, fir_kernel=[1,3,3,1])
        #self.resample=nn.Upsample( scale_factor=(1,S))
    def forward(self,x):
        return self.resample(x) 

class Downsample(nn.Module):
    def __init__(self,S):
        super().__init__()

        self.resample = layerspp.Downsample( with_conv=False, fir=True, fir_kernel=[1,3,3,1])
        #self.resample=nn.AvgPool2d((1,S+1), stride=(1,S), padding=(0,1))
    def forward(self,x):
        return self.resample(x) 
    

class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class FinalBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, 2] 
    Final block. Basiforwardy, a 3x3 conv. layer to map the output features to the output complex spectrogram.

    '''
    def __init__(self, N0):
        super(FinalBlock, self).__init__()
        ksize=(3,3)
        self.conv2=ComplexConv2d(N0,out_channels=1,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect')


    def forward(self, inputs ):

        pred=self.conv2(inputs)

        return pred


class FreqEncodingRFF(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim, N):
        super(FreqEncodingRFF, self).__init__()
        self.N=N
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, N]), requires_grad=False)


        self.f_dim=f_dim #f_dim is fixed
        embeddings=self.build_RFF_embedding()
        self.embeddings=nn.Parameter(embeddings, requires_grad=False) 

        
    def build_RFF_embedding(self):
        """
        Returns:
          table:
              (shape: [C,F], dtype: float32)
        """
        freqs = self.RFF_freq
        #freqs = freqs.to(device=torch.device("cuda"))
        freqs=freqs.unsqueeze(-1) # [1, 32, 1]

        self.n=torch.arange(start=0,end=self.f_dim)
        self.n=self.n.unsqueeze(0).unsqueeze(0)  #[1,1,F]

        table = 2 * np.pi * self.n * freqs

        #print(freqs.shape, x.shape, table.shape)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[1,32,F]
        #print(table.shape)
        table=table.squeeze(0).permute(1,0)
        #print(table.shape)

        return table
    

    def forward(self, input_tensor):

        #print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[2]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.embeddings, [batch_size_tensor, time_dim,self.N*2, self.f_dim])
        fembeddings_2=fembeddings_2.permute(0,2,1,3)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  

class AddFreqEncodingRFF(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim, N):
        super(AddFreqEncodingRFF, self).__init__()
        self.N=N
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, N]), requires_grad=False)


        self.f_dim=f_dim #f_dim is fixed
        embeddings=self.build_RFF_embedding()
        self.embeddings=nn.Parameter(embeddings, requires_grad=False) 

        
    def build_RFF_embedding(self):
        """
        Returns:
          table:
              (shape: [C,F], dtype: float32)
        """
        freqs = self.RFF_freq
        #freqs = freqs.to(device=torch.device("cuda"))
        freqs=freqs.unsqueeze(-1) # [1, 32, 1]

        self.n=torch.arange(start=0,end=self.f_dim)
        self.n=self.n.unsqueeze(0).unsqueeze(0)  #[1,1,F]

        table = 2 * np.pi * self.n * freqs

        #print(freqs.shape, x.shape, table.shape)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[1,32,F]

        return table
    

    def forward(self, input_tensor):

        #print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[-1]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.embeddings, [batch_size_tensor, time_dim,self.N*2, self.f_dim])
        fembeddings_2=fembeddings_2.permute(0,2,3,1)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  
class AddFreqEncoding(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim):
        super(AddFreqEncoding, self).__init__()
        pi=torch.pi
        self.f_dim=f_dim #f_dim is fixed
        n=torch.arange(start=0,end=f_dim)/(f_dim-1)
        # n=n.type(torch.FloatTensor)
        coss=torch.cos(pi*n)
        f_channel = torch.unsqueeze(coss, -1) #(1025,1)
        self.fembeddings= f_channel
        
        for k in range(1,10):   
            coss=torch.cos(2**k*pi*n)
            f_channel = torch.unsqueeze(coss, -1) #(1025,1)
            self.fembeddings=torch.cat((self.fembeddings,f_channel),-1) #(1025,10)

        self.fembeddings=nn.Parameter(self.fembeddings,requires_grad=False)
        #self.register_buffer('fembeddings_const', self.fembeddings)

    

    def forward(self, input_tensor):

        #print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[2]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.fembeddings, [batch_size_tensor, time_dim,2, self.f_dim, 10])
        fembeddings_2=fembeddings_2.permute(0,4,1,3,2)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  #(batch,12,427,1025)


class Decoder(nn.Module):
    '''
    [B, T, F, N] , skip connections => [B, T, F, N]  
    Decoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, unet_args):
        super(Decoder, self).__init__()

        self.Ns=Ns
        self.Ss=Ss
        self.depth=5

        self.dblocks=nn.ModuleList()
        for i in range(self.depth):
            self.dblocks.append(D_Block(layer_idx=i,N0=self.Ns[i+1] ,N=self.Ns[i], S=self.Ss[i],num_tfc=3, ksize=(3,3)))

    def forward(self,inputs, contracting_layers):
        x=inputs
        for i in range(self.depth,0,-1):
            x=self.dblocks[i-1](x, contracting_layers[i-1])
        return x 

class CpxFilm(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim=output_dim
        self.output_layer = nn.Linear(512, 5 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding=sigma_encoding.view(-1, self.output_dim, 5)
        sigma_encoding=sigma_encoding.permute(2,0,1)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        sigma_encoding = sigma_encoding.unsqueeze(-1) #we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        return sigma_encoding[0:3], sigma_encoding[3:5]
class Encoder(nn.Module):

    '''
    [B, T, F, N] => skip connections , [B, T, F, N_4]  
    Encoder side of the U-Net subnetwork.
    '''
    def __init__(self,N0, Ns, Ss, args):
        super(Encoder, self).__init__()
        self.Ns=Ns
        self.Ss=Ss
        self.args=args
        self.depth=args.unet_STFT.depth

        self.contracting_layers = {}

        self.eblocks=nn.ModuleList()
        self.film=nn.ModuleList()

        for i in range(self.depth):
            if i==0:
                Nin=N0
            else:
                Nin=self.Ns[i]

            self.film.append(CpxFilm(Nin))
            self.eblocks.append(E_Block(layer_idx=i,N0=Nin,N01=self.Ns[i],N=self.Ns[i+1],S=self.Ss[i], num_tfc=3, ksize=(3,3)))

        self.i_block=I_Block(self.Ns[self.depth],self.Ns[self.depth],3, (3,3))

    def forward(self, inputs, sigma_encoding):
        x=inputs
        for i in range(self.depth):
            scaleshift = self.film[i](sigma_encoding)
            #apply the modulation here
            gamma , beta = scaleshift

            z=x.clone()
            z[...,0] = (gamma[0])*x[...,0] +(gamma[1])*x[...,1]+ beta[0]
            z[...,1] = (gamma[1])*x[...,0] +(gamma[2])*x[...,1]+ beta[1]
            x=z

            x, x_contract=self.eblocks[i](x)
        
            self.contracting_layers[i] = x_contract #if remove 0, correct this


        x=self.i_block(x)

        return x, self.contracting_layers

class Film(nn.Module):
    def __init__(self, output_dim, bias=True):
        super().__init__()
        self.bias=bias
        if bias:
            self.output_layer = nn.Linear(512, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(512, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        sigma_encoding = sigma_encoding.unsqueeze(-1) #we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma=sigma_encoding
            beta=None

        return gamma, beta

class Gated_residual_layer(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation,
        bias=True
    ):
        super().__init__()
        self.conv= nn.Conv2d(dim,dim,
                  kernel_size=kernel_size,
                  dilation=dilation,
                  stride=1,
                  padding='same',
                  padding_mode='reflect',
                  bias=bias) #freq convolution (dilated) 
        self.act= nn.GELU()
        #self.conv1_1= nn.Conv2d(dim,dim,
        #          kernel_size=1)
        #self.position_gate = nn.Sequential( nn.Linear(64, dim),
        #                                    nn.Sigmoid()) #assuming that 64 is the dimensionality of the RFF freq. positional embeddings
        #self.gn=nn.GroupNorm(8, dim)

    def forward(self, x):
        #gate=self.position_gate(freqembeddings)  #F, N
        #B,N,T,F=x.shape
        #gate = gate.unsqueeze(0).unsqueeze(0) #1,1, F,N
        #gate = gate.permute(0,3,2,1) #1,N,1, F
        #torch.broadcast_to(gate.permute(0,1).unsqueeze(0).unsqueeze(0)
        
        x=(x+self.conv(self.act(x)))/(2**0.5)
        return x
        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=False,
        groups = 8,
        num_layers=8,
        max_dil=8,
        bias=True
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.film=Film(dim, bias=bias)

        self.res_conv = nn.Conv2d(dim, dim_out, 1, padding_mode="reflect", bias=bias) if dim != dim_out else nn.Identity()

        self.H=nn.ModuleList()
        self.num_layers=num_layers

        if self.use_norm:
            self.gnorm=nn.GroupNorm(8,dim)

        self.first_conv=nn.Sequential(nn.GELU(),nn.Conv2d(dim, dim_out,1, bias=bias))

         
        dils2=[2**i for i in range(max_dil)]
        dils1=[1 for i in range(num_layers-max_dil)]
        dils=dils1+dils2
        assert len(dils)==self.num_layers
        
        for i in range(self.num_layers):
            #I use 5x5 kernels instead of 5x3 because I want more attention on the time axis. Anyway, I dont use dilations here in the time axis
            self.H.append(Gated_residual_layer(dim_out, (5,5), (dils[i],1), bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)


    def forward(self, x, sigma):
        
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x=self.gnorm(x)

        if self.bias:
            x=x*gamma+beta
        else:
            x=x*gamma #no bias

        y=self.first_conv(x)

        #freqembeddings=freqembeddings[0:x.shape[-2],:]
        
        for h in self.H:
            y=h(y)

        return (y + self.res_conv(x))/(2**0.5)

class Unet_STFT(nn.Module):

    def __init__(self, args, device):
        super(Unet_STFT, self).__init__()
        self.args=args
        self.depth=args.unet_STFT.depth
        self.embedding = RFF_MLP_Block()
        self.use_norm=True

        #fmax=self.args.sample_rate/2
        #fmin=fmax/(2**self.args.cqt.numocts)
        #fbins=int(self.args.cqt.binsoct*self.args.cqt.numocts) 
        self.device=device
        #self.CQTransform=utils.CQT_cpx(fmin,fbins, self.args.sample_rate, self.args.audio_len, device=self.device, split_0_nyq=False)
        Nin=2

        #self.f_dim=self.args.stft.win_size//2 +1
        self.f_dim=self.args.stft.model.win_size//2
        N_freq_encoding=32
        #self.freqembeddings=FreqEncodingRFF(self.args.stft.win_size//2 +1, N_freq_encoding).embeddings
        self.freq_encoding=AddFreqEncodingRFF(self.f_dim,N_freq_encoding)
        Nin=Nin+N_freq_encoding*2 #hardcoded

        self.use_fencoding=self.args.unet_STFT.use_fencoding

        #Encoder
        self.Ns= args.unet_STFT.Ns
        self.Ss= args.unet_STFT.Ss
        #self.nlayers= [8,7,6,5,4,3]
        self.max_dil= self.args.unet_STFT.nlayers #different nlayers of dilated convolution  per layer as the freq. resolution also changes

        
        #initial feature extractor
        #ksize=(7,7)

        #self.conv2d_1 = nn.Sequential(nn.Conv2d(Nin,self.Ns[0],
        #              kernel_size=ksize,
        #              padding='same',
        #              padding_mode='reflect'),
        ##              nn.ELU())
                        
        self.init_conv= nn.Conv2d(Nin,self.Ns[0],(5,3), padding="same", padding_mode="reflect", bias=False)
        #self.final_conv= nn.Conv2d(self.Ns[0],2,(3,3), padding="same", padding_mode="reflect")

        #initialize last layer with 0


        self.downs=nn.ModuleList([])
        self.middle=nn.ModuleList([])
        self.ups=nn.ModuleList([])
        
        for i in range(self.depth):
            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i-1]
                dim_out=self.Ns[i]

            if i<(self.depth-1):
                self.downs.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, self.use_norm, max_dil=self.max_dil[i], bias=False),
                                            Downsample( self.Ss[i]),
                                            CombinerDown("sum", 2, dim_out, bias=False)]))

            elif i==(self.depth-1): #no downsampling in the last layer
                self.downs.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, self.use_norm, max_dil=self.max_dil[i], bias=False),
                                            ]))

        self.middle.append(nn.ModuleList([
                        ResnetBlock(self.Ns[self.depth], self.Ns[self.depth], self.use_norm, max_dil=self.max_dil[-1], bias=False)
                        ]))
        for i in range(self.depth-1,-1,-1):

            if i==0:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i-1]

            if i>0: 
                self.ups.append(nn.ModuleList(
                                            [
                                            ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, max_dil=self.max_dil[i], bias=False),
                                            Upsample( self.Ss[i]),
                                            CombinerUp("sum", 2, dim_out, bias=False)]))

            elif i==0: #no downsampling in the last layer
                self.ups.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, max_dil=self.max_dil[i], bias=False),
                                            ]))


        self.cropconcat = CropConcatBlock()



    def forward(self, inputs, sigma):
        sigma = self.embedding(sigma)

        inputs=inputs.squeeze(1) #need to squeeze to do the stft, I think
        
        xF =self.do_stft(inputs)
        #print(xF.shape)
        xF=xF.permute(0,3,2,1).contiguous()
        pyr=xF
        #print(xF.shape)

        #xF: B,C(1),T,F,2

        if self.use_fencoding:
            xF=self.freq_encoding(xF)   

        #intitial feature extractor
        #print(xF.shape)
        x=self.init_conv(xF) 
        
        hs=[]
        for i,modules in enumerate(self.downs):
            #print("encoder ", i, x.shape)
           
            if i<(self.depth-1):
                resnet, downsample, combiner=modules
                #print(i,x.shape)
                x=resnet(x,sigma)
                hs.append(x)
                x=downsample(x)
                pyr=downsample(pyr)
                x=combiner(pyr,x)
                

            elif i==(self.depth-1): #no downsampling in the last layer
                (resnet,)=modules
                x=resnet(x,sigma)
                hs.append(x)

        for modules in self.middle:
            #print("middle ", i, x.shape)
            (resnet,) =modules 
            x=resnet(x,sigma)

        pyr=None
        for i,modules in enumerate(self.ups):
            #print("decoder ", i, x.shape)
            j=self.depth -i-1
            if j>0: 
                resnet, upsample, combiner=modules
                
                skip=hs.pop()
                #print(x.shape, skip.shape)
                x =self.cropconcat(x, skip) #there will be problems here, use cropping if necessary
                x=resnet(x,sigma)
                        
                pyr=combiner(pyr,x)
                
                x=upsample(x)
                pyr=upsample(pyr)
                
            elif j==0: #no upsampling in the last layer
                (resnet,)=modules
                skip=hs.pop()
                #print(x.shape, skip.shape)
                x=self.cropconcat(x,skip)
                #x = torch.cat((x, hs.pop()), dim=1)
                x=resnet(x,sigma)
                pyr=combiner(pyr,x)


        #print("end ", x.shape)
        pred=pyr
        #print(pred.shape)
        pred=pred.permute(0,3,2,1)
        pred_time=self.do_istft(pred)
        pred_time=pred_time[:,0:inputs.shape[-1]]
        assert pred_time.shape==inputs.shape, "bad shapes"
        return pred_time

    def do_stft(self,x):
        win_size=self.args.stft.model.win_size
        hop_size=self.args.stft.model.hop_size
        window=torch.hamming_window(window_length=win_size)
        window=window.to(x.device)
        x=torch.cat((x, torch.zeros(x.shape[0],win_size-1).to(x.device)),1)
        stft_signal_noisy=torch.stft(x, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        stft_signal_noisy=stft_signal_noisy.permute(0,2,1,3)
       
        return stft_signal_noisy[:,:,0:-1,:]

    def do_istft(self,x):
        #x=x.squeeze(1)
        x=torch.cat((x,torch.zeros((x.shape[0], x.shape[1],1,x.shape[-1])).to(x.device)),-2) #add zeroes in the nyquist freq.
        win_size=self.args.stft.model.win_size
        hop_size=self.args.stft.model.hop_size
        window=torch.hamming_window(window_length=win_size) #this is slow! consider optimizing
        window=window.to(x.device)
        x=x.permute(0,2,1,3)
        pred_time=torch.istft(x, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)
        return pred_time
            

class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        #print(x1_shape,x2_shape)
        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff),:]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x

