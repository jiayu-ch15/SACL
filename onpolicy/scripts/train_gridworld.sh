#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="merge_local_new_rand_loc_step2000_envs50_ppo3_lr5e-4_grid_scene_asyn"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mapping" --user_name "yang-xy20" --num_agents ${num_agents}\
    --num_obstacles ${num_obstacles} --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' --hidden_size 64 --seed 1 --n_training_threads 1 \
    --n_rollout_threads 2 --num_mini_batch 1 --num_env_steps 50000000 --ppo_epoch 3 --gain 0.01 \
    --lr 5e-4 --critic_lr 5e-4 --max_steps 2000 --use_complete_reward --use_eval --n_eval_rollout_threads 10 --agent_view_size 7 --use_random_pos --use_merge --use_local --grid_size 50
done