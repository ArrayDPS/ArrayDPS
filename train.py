"""

Main script for training
"""
import os
import hydra

import torch
torch.cuda.empty_cache()

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

def run(args):
    """Loads all the modules and starts the training
        
    Args:
      args:
        Hydra dictionary

    """
        
    #some preparation of the hydra args
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    #choose gpu as the device if possible
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirname = os.path.dirname(__file__)

    #define the path where weights will be loaded and saved
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    torch.backends.cudnn.benchmark = True
        
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    # print("Training on: ",args.dset.name)

    #prepare the dataset loader
    import src.dataset_libritts as dataset
    dataset_train=dataset.LIBRITTS_TrainSet(

        root=args.dataset.root,
        # urls=['dev-other'],
        download=False,
        audio_len=args.dataset.audio_len,
        min_audio_len=args.dataset.min_audio_len,
        target_sampling_rate=args.dataset.target_sampling_rate, 
        std_norm=args.dataset.std_norm, 
        std=args.dataset.std
                               
    )
    # import src.dataset_libritts_reverberation as dataset
    # dataset_train=dataset.LIBRITTS_REVERB_TrainSet(

    #     root=args.dataset.root,
    #     rir_root=args.dataset.rir_root,
    #     rir_table_csv_path=args.dataset.rir_table_csv_path,
    #     lower_t60=args.dataset.lower_t60,
    #     upper_t60=args.dataset.upper_t60,
    #     reverb_prob=args.dataset.reverb_prob,

    #     download=False,
    #     audio_len=args.dataset.audio_len,
    #     min_audio_len=args.dataset.min_audio_len,
    #     target_sampling_rate=args.dataset.target_sampling_rate, 
    #     norm=args.dataset.norm
                               
    # )
    train_loader=DataLoader(dataset.LIBRITTS_IterableDataset(dataset_train),num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)
        
    #prepare the model architecture
    
    if args.architecture=="unet_CQT":
        from src.models.unet_cqt import Unet_CQT
        model=Unet_CQT(args, device).to(device)
    elif args.architecture=="unet_STFT":
        from src.models.unet_stft import Unet_STFT
        model=Unet_STFT(args, device).to(device)
    elif args.architecture=="unet_1d":
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

    #prepare the optimizer

    from src.learner import Learner
    
    learner = Learner(
        args.model_dir, model, train_set,  args, log=args.log
    )

    #start the training
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf_libritts_ncsnpp")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()
