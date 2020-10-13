#!/bin/sh
env="ShelterConstruction"
num_agents=1
objective_placement="center"
algo="center_parallel1000_length160_attn"
seed_max=1

echo "env is ${env}, placement is center, num_agents is ${num_agents}, algo is ${algo}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=0 python train_hns_transfertask.py --env_name ${env} --algorithm_name ${algo} --objective_placement ${objective_placement} --num_agents ${num_agents} --seed ${seed_max} --n_rollout_threads 1 --num_mini_batch 1 --episode_length 160 --num_env_steps 100000000 --ppo_epoch 15 --attn
echo "training is done!"
