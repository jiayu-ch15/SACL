#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
      --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed 2 --n_training_threads 1 --n_rollout_threads 1 --render_episodes 1 \
      --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' \
      --model_dir "/home/yangxy/onpolicy/onpolicy/scripts/results/GridWorld/MiniGrid-MultiExploration-v0/rmappo/new_same_loc_step100_envs50_ppo3_lr5e-4_random_grid_scene_goal_noimage_asyn/wandb/run-20210607_141939-ifptifd2/files/" \
      --ifi 0.5 --use_wandb --use_render --max_steps 100 --use_merge
done
