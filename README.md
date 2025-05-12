# ArrayDPS: Unsupervised Blind Speech Separation with a Diffusion Prior

The current repository is the official implementation of the ICML2025 paper:

[ArrayDPS: Unsupervised Blind Speech Separation with A Diffusion Prior](https://arxiv.org/abs/2505.05657) by [Zhongweiyang Xu](https://xzwy.github.io/alanweiyang.github.io/), [Xulin Fan](https://scholar.google.com/citations?user=fU7hjTYAAAAJ&hl=en), [Zhong-Qiu Wang](https://zqwang7.github.io/), [Xilin Jiang](https://scholar.google.com/citations?hl=en&user=FQrkPUwAAAAJ), [Romit Roy Choudhury](https://croy.web.engr.illinois.edu/)

We encourage the readers to look at our demo page at https://arraydps.github.io/ArrayDPSDemo/.

## Setup
We use Python 3.8 and Pytorch 1.11. Other packages are listed in `requirements.txt`. The Environment can be setup by running:
```bash
bash setup.sh
```

Download the speech diffusion model by running:

```bash
link='https://uofi.box.com/shared/static/eent06t4b4hdkjf0vgjzsqw8defa3xbn.pt'
wget -O experiments/raw_WAV_unet_att_8S_3S_8000hz/model_ckpt.pt $link
```

Our paper uses the SMS-WSJ dataset, prepare the dataset following here: [SMS_WSJ GitHub repository](https://github.com/fgnt/sms_wsj).

## Separation
To separate and evaluate on SMS-WSJ test dataset, run the following command with a GPU with >7GB of cuda memory. Also remember to set  `root_dir` as the SMS-WSJ dataset directory (containing early, reverb, observation...). The separation results are all saved in `./separation_outputs`.
```bash
python separate.py \
  --diffusion_model_type 'anechoic' \
  --config_path 'conf/conf_libritts_unet1d_attention_8k.yaml' \
  --architecture 'unet_1d_att' \
  --checkpoint 'model_ckpt.pt' \
  --root_dir 'smswsj_dataset_dir_(containing early, reverb, observation.....)' \
  --num_speakers 2 \
  --reverb 1 \
  --n_channels 3 \
  --num_steps 400 \
  --max_trials 5 \
  --snr_stop 35 \
  --max_trials2 5 \
  --snr_stop2 14 \
  --max_trials3 5 \
  --snr_stop3 10 \
  --sigma_max 0.8 \
  --rho 10 \
  --schurn 30 \
  --xi 2.0 \
  --n_fft 512 \
  --hop_length 128 \
  --lambda_reg 0.002 \
  --n_frames_past 13 \
  --n_frames_future 0 \
  --fcp_epsilon 0.001 \
  --ref_loss_weight 0.3 \
  --ref_loss_snr_threshold 20 \
  --ref_loss_max_step 200 \
  --use_warm_initialization 1 \
  --warm_initialization_rescale 1 \
  --warm_initialization_sigma 0.057 \
  --initialized_filter_step 100 \
  --save_dir "./separation_outputs" \
  --n_samples 1332 \
  --blind 1 \
  --start_sample 0
```


## Diffusion prior model training
To train an unsupervised speech diffusion model, first download LibriTTS dataset in [https://openslr.org/60/](https://openslr.org/60/).

Update your wandb log information and LibriTTS dataset path in `conf/conf_libritts_unet1d_attention_8k.yaml`

To start training, run
```bash
bash start_libritts_wav_att.sh
```

## Citations
If you use our model for your research, please consider citing
```
@misc{xu2025unsupervisedblindspeechseparation,
      title={ArrayDPS: Unsupervised Blind Speech Separation with a Diffusion Prior}, 
      author={Zhongweiyang Xu and Xulin Fan and Zhong-Qiu Wang and Xilin Jiang and Romit Roy Choudhury},
      year={2025},
      eprint={2505.05657},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.05657}, 
}
```