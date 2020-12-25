#!/bin/sh
env="highway"
scenario="highway-v0"
num_agents=3
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=4 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --use_render --episode_length 25 --use_wandb
done
