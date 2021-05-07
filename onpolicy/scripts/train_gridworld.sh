#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="dirrew_encoder_3predator_1prey_0cobstacle_alpha0.1_discounter0.1_maxstep200_v3"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "human" --user_name "zoeyuchao" --num_agents ${num_agents}\
    --num_obstacles ${num_obstacles} --cnn_layers_params '32,3,1,1' --seed 1 --n_training_threads 1 \
    --n_rollout_threads 128 --num_mini_batch 1 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4
done
