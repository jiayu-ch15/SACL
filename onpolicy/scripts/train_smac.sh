#!/bin/sh
env="StarCraft2"
map="corridor"
algo="rmappg"
exp="debug"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 10 --num_env_steps 50000000 --ppo_epoch 15 --use_wandb
    echo "training is done!"
done