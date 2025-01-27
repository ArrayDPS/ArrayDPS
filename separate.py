import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
torch.cuda.empty_cache() 

import yaml
from pathlib import Path
from dotmap import DotMap
import argparse

import torchaudio
import csv
from sdr import batch_SDR_torch
import numpy as np

def save_separated_samples(outs_list, save_dir, sample_idx):
    """
    Saves the separated audio samples into a specific directory.
    
    Args:
    - outs (torch.Tensor): a list of separated audio samples of shape (n_spk, T).
    - save_dir (str): The base directory where the samples will be saved.
    - sample_idx (int): The index of the current sample.
    """
    # Create a folder for the current sample
    sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save each separated source as s1.wav, s2.wav, ..., s_n_spk.wav
    for trial_idx, outs in enumerate(outs_list):
        trial_dir = os.path.join(sample_dir, f"trial_{trial_idx}")
        os.makedirs(trial_dir, exist_ok=True)
        n_spk, T = outs.shape
        for spk in range(n_spk):
            save_path = os.path.join(trial_dir, f"s{spk + 1}.wav")
            torchaudio.save(save_path, outs[spk].unsqueeze(0).cpu(), sample_rate=8000)  # Assuming 16kHz sample rate

def save_sdr_list(sdr_list, mix_snr_list, save_dir, max_trials):
    """
    Saves the list of SDRs and mix SNRs to a CSV file.

    Args:
    - sdr_list (list of tuples): List of (sample_index, list[SDR_value]) tuples.
    - mix_snr_list (list of tuples): List of (sample_index, list[mix_SNR_value]) tuples.
    - save_dir (str): The directory where the CSV will be saved.
    """
    sdr_csv_path = os.path.join(save_dir, "sdr_mix_snr.csv")
    
    with open(sdr_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        title = ["Sample Index"]+[f"SDR{trial_idx}" for trial_idx in range(max_trials)]+[f"Mix SNR {trial_idx}" for trial_idx in range(max_trials)] + ['max_sdr', 'max_mix_snr']
        writer.writerow(title)
        # Combine SDR and Mix SNR lists into a single row per sample
        for (sample_index, sdr_values), (_, mix_snr_values) in zip(sdr_list, mix_snr_list):
            sdr_values = sdr_values + [np.nan for _ in range(max_trials-len(sdr_values))]
            mix_snr_values = mix_snr_values + [np.nan for _ in range(max_trials-len(mix_snr_values))]
            max_sdr = max(sdr_values)
            max_mix_snr = max(mix_snr_values)
            writer.writerow([sample_index]+sdr_values+mix_snr_values+[max_sdr, max_mix_snr])

def check_existing_outputs(save_dir, sample_idx, num_speakers, max_trials):
    """
    Checks if the output directories and files for a given sample already exist.

    Args:
    - save_dir (str): The directory where outputs are saved.
    - sample_idx (int): The index of the sample to check.
    - num_speakers (int): The number of speakers to check for each audio file.
    - max_trials (int): The maximum number of trials to check.

    Returns:
    - bool: True if all expected outputs exist, False otherwise.
    """
    sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
    if not os.path.exists(sample_dir):
        return False

    for trial_idx in range(max_trials):
        trial_dir = os.path.join(sample_dir, f"trial_{trial_idx}")
        if not os.path.exists(trial_dir):
            return False

        expected_files = [f"s{spk + 1}.wav" for spk in range(num_speakers)]
        actual_files = os.listdir(trial_dir)
        
        if not all(file in actual_files for file in expected_files):
            return False

    return True

def run_inference(dataset, sampler, device, save_dir, args_configurable, n_samples, start_sample=0):
    """
    Runs inference on the dataset and saves the separated audio samples and SDRs.

    Args:
    - dataset: The dataset to process.
    - sampler: The sampler used for separation and spatialization.
    - device: The device to use for computation (e.g., 'cuda' or 'cpu').
    - save_dir: The directory where separated samples will be saved.
    - args_configurable: Configuration object containing various settings.
    - n_samples: The maximum number of samples to process.
    """
    sdr_list = []  # List to store SDR values for each sample
    mixture_snr_list = []

    for i, (mixture, early, tail) in enumerate(dataset):
        
        if check_existing_outputs(save_dir, i, args_configurable.num_speakers, 5):
            continue
        
        sources = early + tail
        if i < start_sample:
            continue
        if i - start_sample > n_samples:
            break
        
        # if i >= n_samples:  # Break the loop if the number of samples exceeds n_samples
        #     break
        # if i <= 2:
        #     continue

        if args_configurable.architecture=='unet_1d_att':
            # need to pad the signal so the length is a power of 2
            sig_len = mixture.shape[-1]
            target_len = int(2**np.ceil(np.log2(sig_len)))
            mixture = torch.nn.functional.pad(mixture, (0, target_len - sig_len))
            sources = torch.nn.functional.pad(sources, (0, target_len - sig_len))
            
        # Ensure the mixture and sources are properly moved to the device
        input = mixture.unsqueeze(0).to(device)  # (1, n_ch, T)
        sources = sources.to(device)  # (n_spk, n_ch, T)

        # # Calculate steering vector if not blind
        # w_steer_vec = None
        # if not args_configurable.blind:
        #     ref_source = sources[:, 0, :].unsqueeze(0).to(device)  # 1, n_spk, T
        #     non_ref_mixtures = mixture[1:, :].unsqueeze(0).to(device)  # 1, n_ch - 1, T
        #     w_steer_vec, sources_hat_oracle_ref = sampler.spatialization(ref_source, non_ref_mixtures)  # 1, n_spk, n_ch-1, T

        # Perform separation with max_trials
        current_sample_sdrs = []
        current_sample_mixture_snrs = []
        current_sample_outs = []
        max_snr = -np.inf
        # for trial_idx in range(args_configurable.max_trials):
        n_trials = 0
        while(1): # for 
            outs = sampler.separate(input, args_configurable.num_speakers, device)  # 1, n_spk, T

            if args_configurable.architecture=='unet_1d_att':
                outs = outs[..., :sig_len]
                mixture = mixture[..., :sig_len]
                sources = sources[..., :sig_len]
            
            # spatialize to map to reference channel
            separated_sources = outs.reshape(1, args_configurable.num_speakers, -1) # 1, n_spk, T
            w, separated_sources_mc = sampler.spatialization(separated_sources, mixture.unsqueeze(0).to(device)) # 1, n_spk, n_channel, T
            
            
            # calcualte sdr and mix snr
            sdr = batch_SDR_torch(separated_sources_mc[:,:,0,:], sources[:, 0, :].unsqueeze(0))
            mix_snr = sampler.spatialization.calc_snr(mixture[0], separated_sources_mc[0, :, 0, :].sum(0).cpu())
            print(f"Sample {i} trial {n_trials} SDR: {sdr} MIXSNR: {mix_snr}")
            
            # add output, add si-sdr, add mix_snr
            current_sample_outs.append(outs.squeeze(0))
            current_sample_sdrs.append(sdr.item())
            current_sample_mixture_snrs.append(mix_snr.item())
            
            # update max_snr
            max_snr = sdr.item() if sdr.item() > max_snr else max_snr
            
            # max_trials', type=int, default=5, help='max trials needed when snr in [snr_stop2, snr_stop]')
            # snr_stop', type=float, default=18., help='snr threshold (no need to trial more beyond this snr)')           
            # max_trials2', type=int, default=15, help='max trials needed when snr in [snr_stop3, snr_stop2]')
            # snr_stop2', type=float, default=10., help='snr threshold')          
            # max_trials3', type=int, default=25, help='max trials needed when snr < snr_stop3')
            # snr_stop3', type=float, default=5., help='snr threshold')
            n_trials += 1
            if max_snr >= args_configurable.snr_stop:
                break
            elif max_snr >= args_configurable.snr_stop2 and n_trials >= args_configurable.max_trials: # [snr_stop2, snr_stop]
                break
            elif max_snr >= args_configurable.snr_stop3 and n_trials >= args_configurable.max_trials2: # [snr_stop3, snr_stop2]
                break
            elif n_trials >= args_configurable.max_trials3: # already finished lots of trials and still not working
                break
            
            
        # Save separated samples
        save_separated_samples(current_sample_outs, save_dir, i)

        # Compute SDR and store it in the list
        sdr_list.append((i, current_sample_sdrs))  # Store sample index and SDR value
        mixture_snr_list.append((i, current_sample_mixture_snrs))
    # Save the SDR list to a CSV file
    save_sdr_list(sdr_list, mixture_snr_list, save_dir, args_configurable.max_trials3)

def modify_and_create_save_dir(config):
    """
    Modifies the save_dir based on the configuration and creates the necessary directories.

    Args:
    - config (dict): The configuration dictionary containing various settings.
    """
    # Extract values from the config dictionary
    num_speakers = config['num_speakers']
    reverb = 'reverb' if config['reverb'] else 'anechoic'
    architecture = config['architecture']
    blind_or_oracle = 'blind' if config['blind'] else 'oracle'

    # Modify the save_dir based on the logic
    folder_name = concatenate_params(args_dict)
    new_save_dir = os.path.join(
        config['save_dir'],
        f"{num_speakers}speaker_{reverb}",
        architecture,
        blind_or_oracle,
        folder_name
    )

    # Update the save_dir in the config dictionary
    config['save_dir'] = new_save_dir

    # Create the directory if it doesn't exist
    os.makedirs(new_save_dir, exist_ok=True)

    print(f"Modified and created directory: {new_save_dir}")

def concatenate_params(args_dict):
    """
    Concatenates specific parameters from a dictionary into a formatted string.

    Args:
    - args_dict (dict): The dictionary containing the parameters.

    Returns:
    - str: The formatted string.
    """
    n_channels = args_dict['n_channels']
    num_steps = args_dict['num_steps']
    
    max_trials = args_dict['max_trials']
    snr_stop = args_dict['snr_stop']
    
    max_trials2 = args_dict['max_trials2']
    snr_stop2 = args_dict['snr_stop2']
    
    max_trials3 = args_dict['max_trials3']
    snr_stop3 = args_dict['snr_stop3']
    
    sigma_min = args_dict['sigma_min']
    sigma_max = args_dict['sigma_max']
    rho = args_dict['rho']
    schurn = args_dict['schurn']
    xi = args_dict['xi']
    n_fft = args_dict['n_fft']
    hop_length = args_dict['hop_length']
    lambda_reg = args_dict['lambda_reg']
    n_frames_past = args_dict['n_frames_past']
    n_frames_future = args_dict['n_frames_future']
    fcp_epsilon = args_dict['fcp_epsilon']
    ref_loss_weight = args_dict['ref_loss_weight']
    ref_loss_snr_threshold = args_dict['ref_loss_snr_threshold']

    # parameters for iva initialization    
    ref_loss_max_step = args_dict['ref_loss_max_step']
    use_warm_initialization = args_dict['use_warm_initialization']
    warm_initialization_rescale = args_dict['warm_initialization_rescale']
    warm_initialization_sigma = args_dict['warm_initialization_sigma']
    initialized_filter_step = args_dict['initialized_filter_step']

    diffusion_model_type = args_dict['diffusion_model_type']

    # Create the concatenated string with all parameters
    concatenated_string = (
        f"{n_channels}channels_{num_steps}steps_{max_trials}_{max_trials2}_{max_trials3}maxtrials_snrstop_{snr_stop}_{snr_stop2}_{snr_stop3}_sigma{sigma_min}-{sigma_max}_rho{rho}_diff{diffusion_model_type}_"
        f"schurn{schurn}_xi{xi}_nfft{n_fft}_hop{hop_length}_lambda{lambda_reg}_"
        f"p{n_frames_past}_f{n_frames_future}_fcpeps{fcp_epsilon}_"
        f"ref_l_w{ref_loss_weight}_ref_l_snr_th{ref_loss_snr_threshold}_ref_l_max_step{ref_loss_max_step}_"
        f"warm_init{use_warm_initialization}_{warm_initialization_rescale}_{warm_initialization_sigma}_init_filter_step{initialized_filter_step}"
    )

    return concatenated_string


parser = argparse.ArgumentParser(description="separator argument parser")

# which model to use
parser.add_argument('--diffusion_model_type', type=str, default='anechoic', help='reverb or anechoic')

parser.add_argument('--config_path', type=str, default='./conf/conf_libritts_unet1d_attention_8k.yaml', help='Path to the config file.')
parser.add_argument('--architecture', type=str, default='unet_1d', help='model architecture (unet_1d, ncsnpp, unet_1d_att)')
parser.add_argument('--checkpoint', type=str, default='weights-459999.pt', help='Path to the checkpoint file.')

# dataset configs
parser.add_argument('--root_dir', type=str, default='your_smswsj_dataset_path', help='dataset path, smswsj')
parser.add_argument('--num_speakers', type=int, default=2, help='number of speakers, 2 or 3')
parser.add_argument('--reverb', type=lambda x: bool(int(x)), default=True, help='reverberation or not')
parser.add_argument('--n_channels', type=int, default=4, help='number of channels to use for RG')

# sampling configs
parser.add_argument('--num_steps', type=int, default=350, help='number of diffusion steps')
parser.add_argument('--max_trials', type=int, default=2, help='max trials needed when snr in [snr_stop2, snr_stop]')
parser.add_argument('--snr_stop', type=float, default=18., help='snr threshold (no need to trial more beyond this snr)')

parser.add_argument('--max_trials2', type=int, default=3, help='max trials needed when snr in [snr_stop3, snr_stop2]')
parser.add_argument('--snr_stop2', type=float, default=13., help='snr threshold')

parser.add_argument('--max_trials3', type=int, default=4, help='max trials needed when snr < snr_stop3')
parser.add_argument('--snr_stop3', type=float, default=10., help='snr threshold')

parser.add_argument('--sigma_min', type=float, default=0.0001, help='minimum noise level')
parser.add_argument('--sigma_max', type=float, default=8, help='maximum noise level')
parser.add_argument('--rho', type=int, default=10, help='scheduling parameter as in EDM')
parser.add_argument('--schurn', type=float, default=30, help='stochasticity parameter as in EDM')
parser.add_argument('--xi', type=float, default=1.3, help='xi weighting for RG')

# parameters for relative channel estimation (later let's consider multi resolution)
parser.add_argument('--n_fft', type=int, default=512, help='n_fft')
parser.add_argument('--hop_length', type=int, default=128, help='hop_length')
parser.add_argument('--lambda_reg', type=float, default=1e-3, help='lambda_reg')

parser.add_argument('--n_frames_past', type=int, default=20, help='number of past frames for steer filtering')
parser.add_argument('--n_frames_future', type=int, default=0, help='number of future frames for steer filtering')

parser.add_argument('--fcp_epsilon', type=float, default=1e-2, help='fcp epsilon')

# parameters for reference loss
parser.add_argument('--ref_loss_weight', type=float, default=0.3, help='weight of reference channle reconstruction guidance (direct)')
parser.add_argument('--ref_loss_snr_threshold', type=float, default=20., help='beyond this snr, no need to optimize for reference channel reconstruction loss')
parser.add_argument('--ref_loss_max_step', type=int, default=100, help='no reference is allowed when step >= ref_loss_max_step')

# parameters for iva initialization
parser.add_argument('--use_warm_initialization', type=lambda x: bool(int(x)), default=True, help='whether use IVA out for source initialization')
parser.add_argument('--warm_initialization_rescale', type=lambda x: bool(int(x)), default=False, help='whether rescale the iva output')
parser.add_argument('--warm_initialization_sigma', type=float, default=0.057, help='warm initialization sigma')
parser.add_argument('--initialized_filter_step', type=int, default=200, help='for the first initialized_filter_step steps, using the iva initialized filter')



# save configurations
parser.add_argument('--save_dir', type=str, help='path to save')
parser.add_argument('--n_samples', type=int, default=2,  help='number of samples to inference, maximize 3000')
parser.add_argument('--blind', type=lambda x: bool(int(x)), default=True, help='blind source separation (1 for True, 0 for False)')

parser.add_argument('--start_sample', type=int, default=0, help='the sample to start inference')

# set up arguments
args_configurable = parser.parse_args()

# setup save_dir and save configurations
args_dict = vars(args_configurable)
modify_and_create_save_dir(args_dict)
args_configurable.save_dir = args_dict['save_dir']
import json
with open(os.path.join(args_configurable.save_dir, 'config.json'), 'w') as json_file:
    json.dump(args_dict, json_file, indent=4)

args = yaml.safe_load(Path(args_configurable.config_path).read_text()) # raw wav longer
args = DotMap(args)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dirname = os.getcwd()
args.model_dir = os.path.join(dirname, str(args.model_dir))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dirname = os.getcwd()

#define the path where weights will be loaded and audio samples and other logs will be saved
args.model_dir = os.path.join(dirname, str(args.model_dir))
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

args.architecture=args_configurable.architecture
args.inference.checkpoint=args_configurable.checkpoint
args.sample_rate=16000
args.diffusion_parameters.sigma_data=0.057

args.inference.T=args_configurable.num_steps
args.diffusion_parameters.sigma_min=args_configurable.sigma_min
args.diffusion_parameters.sigma_max=args_configurable.sigma_max
args.diffusion_parameters.ro=args_configurable.rho
args.inference.xi=args_configurable.xi


# load model
from src.utils.setup import load_ema_weights

if args.architecture=="unet_1d":
    from src.models.unet_1d import Unet_1d
    model=Unet_1d(args, device).to(device)
elif args.architecture=="ncsnpp":
    from src.models.ncsnpp import NCSNppTime
    model = NCSNppTime(args).to(device)
elif args.architecture=="unet_1d_att":
    from src.models.unet_wav import UNet1d
    model = UNet1d(args.unet_wav, device).to(device)
else:
    raise NotImplementedError
model=load_ema_weights(model,os.path.join(args.model_dir, args.inference.checkpoint))
print(f'finish load model from {os.path.join(args.model_dir, args.inference.checkpoint)}')

# instantiate dataset
fs=8000
from src.dataset_sms_wsj import SMSWSJDataset
dataset = SMSWSJDataset(
    smswsj_dir=args_configurable.root_dir,
    sample_rate=fs,
    num_channels=args_configurable.n_channels
)

# instantiate sampler and SDE
from src.sampler_spatial_v1_reverb_iva_8kHz import Sampler
# # instantiate sampler and SDE
# if args_configurable.wiener_version == 'v0':
#     from src.sampler_spatial import Sampler
# elif args_configurable.wiener_version == 'v1':
#     from src.sampler_spatial_v1 import Sampler
# elif args_configurable.wiener_version == 'v2':
#     from src.sampler_spatial_v2 import Sampler
# else:
#     AssertionError
    
from src.sde import  VE_Sde_Elucidating

diff_parameters = VE_Sde_Elucidating(args.diffusion_parameters, args.diffusion_parameters.sigma_data)
sampler=Sampler(
    model, 
    diff_parameters, 
    args, 
    args.inference.xi, 
    order=2,
    n_fft=args_configurable.n_fft, 
    hop_length=args_configurable.hop_length,
    win_length=args_configurable.n_fft,
    lambda_reg = args_configurable.lambda_reg,
    n_frames_past=args_configurable.n_frames_past,
    n_frames_future=args_configurable.n_frames_future,
    fcp_epsilon=args_configurable.fcp_epsilon,
    
    n_spks=args_configurable.num_speakers, # new, but handled
    

    use_warm_initialization=args_configurable.use_warm_initialization, # new
    warm_initialization_rescale=args_configurable.warm_initialization_rescale, # new
    warm_initialization_sigma=args_configurable.warm_initialization_sigma, # new
    initialized_filter_step=args_configurable.initialized_filter_step, # new
    
    
    ref_loss_weight=args_configurable.ref_loss_weight,
    ref_loss_snr_threshold=args_configurable.ref_loss_snr_threshold, # new, but already handled
    ref_loss_max_step=args_configurable.ref_loss_max_step, # new
    )

run_inference(dataset, sampler, device, args_configurable.save_dir, args_configurable, n_samples=args_configurable.n_samples, start_sample=args_configurable.start_sample)