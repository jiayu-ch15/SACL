# SACL

## 1. Install

### 1.1 instructions

   test on CUDA == 10.1   

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   cd onpolicy
   pip install -e . 
   pip install wandb icecream setproctitle gym seaborn tensorboardX slackweb psutil slackweb pyastar2d einops
```

### 1.2 hyperparameters

* config.py: contains all hyper-parameters

* default: use GPU, chunk-version recurrent policy and shared policy

* other important hyperparameters:
  - use_centralized_V: Centralized training (MA) or Centralized training (I)
  - use_single_network: share base or not
  - use_recurrent_policy: rnn or mlp
  - use_eval: turn on evaluation while training, if True, u need to set "n_eval_rollout_threads"
  - wandb_name: For example, if your wandb link is https://wandb.ai/mapping, then you need to change wandb_name to "mapping". 
  - user_name: only control the program name shown in "nvidia-smi".

## 2. Usage

### 2.1 MPE

``` Bash
   conda activate marl
   cd scripts
   bash train_mpe_ensemble_curriculum.sh # sacl
   bash train_mpe.sh # self-play
   bash train_mpe_br.sh # obtain the approximate exploitability
```

### 2.2 Google Research Football

see https://github.com/google-research/football readme.

``` Bash
   conda activate marl
   cd scripts
   bash train_football_curriculum.sh # sacl
   bash train_football.sh # self-play
   bash train_football_br.sh # obtain the approximate exploitability
```