#!/bin/sh
env="Highway"
scenario="highway-v0"
num_agents=3
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=4 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 20 --episode_length 25 --log_interval 1 --use_wandb
done
