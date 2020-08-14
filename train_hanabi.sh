#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="full_2_128"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 200 --num_mini_batch 1 --episode_length 80 --num_env_steps 10000000 --ppo_epoch 20 --use_clipped_value_loss --cuda --hidden_size 128
    echo "training is done!"
done
