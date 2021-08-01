#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="single_merge_local_time_penalty_partial_reward55_new_rand_loc_step100_envs50_size19_ppo3_lr5e-4_grid_scene_asyn"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mapping" --user_name "yang-xy20" --num_agents ${num_agents}\
    --num_obstacles ${num_obstacles} --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' --hidden_size 64 --seed 2 --n_training_threads 1 \
    --n_rollout_threads 50 --num_mini_batch 1 --num_env_steps 50000000 --ppo_epoch 3 --gain 0.01 \
    --lr 5e-4 --critic_lr 5e-4 --max_steps 100 --use_complete_reward --use_eval --n_eval_rollout_threads 10\
    --agent_view_size 7 --use_random_pos --use_local --grid_size 19 --use_single --use_partial_reward --use_time_penalty --use_merge
done