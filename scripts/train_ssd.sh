#!/bin/sh
env="Harvest"
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_ssd.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --ppo_epoch 5 --episode_length 100 --num_env_steps 1000000 --use_wandb
    echo "training is done!"
done
