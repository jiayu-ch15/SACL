#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Very-Small"
num_agents=2
algo="very_small_test"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 200 --num_mini_batch 1 --episode_length 80 --num_env_steps 10000000 --ppo_epoch 15 --use_clipped_value_loss --cuda --hidden_size 128 --data_chunk_length 20
    echo "training is done!"
done
