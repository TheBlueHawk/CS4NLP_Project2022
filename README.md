This repository allows to reproduce the experiments of the CS4NLP 2022 project of group 1 
vat-pytorch: A Plug-and-Play library for Virtual Adversarial Training.
# Installation instructions
Our repo is tested on Python 3.9
## Local installation
1. Install torch, torchvision, torchtext, torchmetrics following the instructions on https://pytorch.org/get-started/locally
2. `pip install -r requirements.txt`

## Euler setup
```bash
env2lmod
module load gcc/8.2.0
module load python_gpu/3.8.5
module load eth_proxy 
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt
``` 

# Reproduce final experiments
To reproduce the final runs over 10 seeds for each model and dataset, you can use the following Weight&Biases command from the root folder:
```bash
wandb login
wandb sweep final_<model>_<dataset>
``` 
This will produce a `<sweep_ID>` that you can use to distribute the 10 runs on one or more machines using:
```bash
wandb login
wandb agent `<sweep_ID>`
``` 

# Run arbitrary experiments
You can just run the command `python vat_hp_tune.py` manually passing all the desired arguments like `--pretrained-model`, `--vat`, `--lr`, `--vat-loss-weight` ... More details on the available arguments can be found in the argument parser at the end of the `vat_hp_tune.py` file. If possible, we recommand to set `--precision` to 16, to allow mixed-precision training and obtain significant speedup. 

# Acknowledgement
MCTACO paper, that we use to replicate the F1 and exact match metrics:
```
@inproceedings{ZKNR19,
    author = {Ben Zhou, Daniel Khashabi, Qiang Ning and Dan Roth},
    title = {“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding },
    booktitle = {EMNLP},
    year = {2019},
}
```



We used Weights & Biases for experiment tracking and visualizations to develop insights for this paper.
```
@misc{wandb,
title = {Experiment Tracking with Weights and Biases},
year = {2020},
note = {Software available from wandb.com},
url={https://www.wandb.com/},
author = {Biewald, Lukas},
}
```

We used PyTorch Lightning, a lightweight PyTorch wrapper for high-performance AI research to organize and scale our models without all the associated boilerplate. 
```
@article{falcon2019pytorch,
  title={Pytorch lightning},
  author={Falcon, William and others},
  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning},
  volume={3},
  number={6},
  year={2019}
}
```

We used the transformers library from huggingface (https://github.com/huggingface/transformers) for all of our transformer based models.


