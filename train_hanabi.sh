#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
#algo="ff_fullminimal2_parallel2000_length80_batch12_ppo15_lr7e-4_512_2layers_relu_100M"
algo="test"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 1 --num_mini_batch 1 --episode_length 5 --num_env_steps 100000000 --ppo_epoch 15 --lr 7e-4 --hidden_size 512 --use_ReLU --layer_N 2 --eval
    echo "training is done!"
done
