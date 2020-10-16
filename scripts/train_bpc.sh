#!/bin/sh
env="BlueprintConstruction"
scenario_name="empty"
num_agents=2
num_boxes=4
#algo="empty_grab_centersite1_box1_parallel500_length160_attn"
algo="test"
seed_max=1

echo "env is ${env}, scenario is ${scenario_name}, num_agents is ${num_agents}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_hns_transfertask.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 150 --num_env_steps 100000000 --ppo_epoch 15 --gain 1 --attn --hidden_size 128 --attn_size 128
    echo "training is done!"
done
