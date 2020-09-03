#!/bin/sh
env="ShelterConstruction"
num_agents=1
algo="center_parallel1000_length160_attn"
seed_max=1

echo "env is ${env}, placement is center, num_agents is ${num_agents}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train_hns_transfertask.py --env_name ${env} --algorithm_name ${algo} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 1 --num_mini_batch 1 --episode_length 160 --num_env_steps 100000000 --ppo_epoch 15 --attn
    echo "training is done!"
done
