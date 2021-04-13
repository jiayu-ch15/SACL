#!/bin/sh
env="Human"
scenario="simple_tag"
num_landmarks=2
num_good_agents=1
num_adversaries=3
algo="rmappo"
exp="train_adversaries_shape"
seed_max=1
ulimit -n 234567

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_human.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed 1 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 200000000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "zoeyuchao"
done