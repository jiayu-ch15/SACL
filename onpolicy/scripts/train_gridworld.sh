#!/bin/sh
env="GridWorld"
scenario="MiniGrid-Human-v0"
num_agents=3
num_preies=1
num_obstacles=0
algo="rmappo"
exp="dirrew_encoder_3predator_1prey_0obstacle_alpha0.02_discounter0.1"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_gridworld.py --use_direction_reward --direction_alpha 0.02 --coverage_discounter 0.1 \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "human" --user_name "huangruixin" --num_agents ${num_agents} --num_preies ${num_preies} \
    --num_obstacles ${num_obstacles} --cnn_layers_params '32,3,1,1' --seed 1 --n_training_threads 1 \
    --n_rollout_threads 128 --num_mini_batch 1 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4  
done
