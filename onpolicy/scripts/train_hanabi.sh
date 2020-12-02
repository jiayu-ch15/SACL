#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
algo="rmappo"
exp="debug"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1000 --n_eval_rollout_threads 32 --num_mini_batch 2 --episode_length 80 --num_env_steps 100000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --hidden_size 512 --layer_N 2 --use_eval
    echo "training is done!"
done
