#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
algo="test"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 2 --num_mini_batch 1 --episode_length 10 --num_env_steps 100000000 --ppo_epoch 2
    echo "training is done!"
done
