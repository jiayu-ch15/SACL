#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=3
algo="small3_oneeval_parallel200_length50_512_valueloss"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 200 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000000 --ppo_epoch 15 --hidden_size 512
    echo "training is done!"
done
