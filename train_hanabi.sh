#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=4
algo="small4_alleval_parallel100_length50_512"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 100 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000000 --ppo_epoch 15 --hidden_size 512
    echo "training is done!"
done
