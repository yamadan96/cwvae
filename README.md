# Clockwork Variational Autoencoders (CW-VAE)

Vaibhav Saxena, Jimmy Ba, Danijar Hafner

<img src="https://danijar.com/asset/cwvae/header.gif">

If you find this code useful, please reference in your paper:

```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```

## Method

Clockwork VAEs are deep generative model that learn long-term dependencies in video by leveraging hierarchies of representations that progress at different clock speeds. In contrast to prior video prediction methods that typically focus on predicting sharp but short sequences in the future, Clockwork VAEs can accurately predict high-level content, such as object positions and identities, for 1000 frames.

Clockwork VAEs build upon the [Recurrent State Space Model (RSSM)](https://arxiv.org/pdf/1811.04551.pdf), so each state contains a deterministic component for long-term memory and a stochastic component for sampling diverse plausible futures. Clockwork VAEs are trained end-to-end to optimize the evidence lower bound (ELBO) that consists of a reconstruction term for each image and a KL regularizer for each stochastic variable in the model.

More information:

- [Research paper (PDF)](https://arxiv.org/pdf/2102.09532.pdf)
- [Project website](http://danijar.com/cwvae)

## Instructions

This repository contains the code for training the Clockwork VAE model on the datasets `minerl`, `mazes`, and `mmnist`.

The datasets will automatically be downloaded into the `--datadir` directory.

```sh
python3 train.py --logdir /path/to/logdir --datadir /path/to/datasets --config configs/<dataset>.yml 
python3 train.py --logdir logs/ --datadir minerl_navigate/ --config configs/minerl.yml 
python3 train_oops.py --logdir logs/ --datadir datadir/ --config configs/oops.yml
```

The evaluation script writes open-loop video predictions in both PNG and NPZ format and plots of PSNR and SSIM to the data directory.

```sh
python3 eval.py --logdir /path/to/logdir
python3 eval.py --logdir logs/minerl/minerl_cwvae_rssmcell_3l_f6_decsd0.4_enchl3_ences1000_edchnlmult1_ss100_ds800_es800_seq100_lr0.0001_bs50/model
```
