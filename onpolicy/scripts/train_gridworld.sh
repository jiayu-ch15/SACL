#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="merge_ppo5"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mapping" --user_name "zoeyuchao" --num_agents ${num_agents}\
    --num_obstacles ${num_obstacles} --cnn_layers_params '32,3,1,1' --hidden_size 64 --seed 1 --n_training_threads 1 \
    --n_rollout_threads 50 --num_mini_batch 1 --num_env_steps 50000000 --ppo_epoch 5 --gain 0.01 \
    --lr 5e-4 --critic_lr 5e-4 --max_steps 100 --use_complete_reward --use_merge
done
