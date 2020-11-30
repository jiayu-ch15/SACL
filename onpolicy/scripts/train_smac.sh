#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmappo"
exp="debug_single"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 20 --num_env_steps 50000000 --ppo_epoch 15 --use_eval --use_value_active_masks --use_wandb --use_single_network --share_policy
    echo "training is done!"
done