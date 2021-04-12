#!/bin/sh
env="Human"
scenario="simple_tag"
num_landmarks=2
num_good_agents=1
num_adversaries=3
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train_human.py --use_eval --use_wandb --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed 1 --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 50 --num_mini_batch 1 --episode_length 25 --num_env_steps 200000000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "zoeyuchao" --prey_model_dir "/home/yuchao/project/onpolicy/onpolicy/envs/mpe/pretrained_models/"
done
