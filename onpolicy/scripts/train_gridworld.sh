#!/bin/sh
env="GridWorld"
scenario="MiniGrid-DoorKey-5x5-v0"
num_agents=1
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gridworld.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --use_wandb --user_name "zoeyuchao" --num_agents ${num_agents} --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4
done
