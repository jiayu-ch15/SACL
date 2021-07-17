#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="single_merge_local_new_rand_loc_step150_envs30_size31_ppo3_lr5e-4_grid_scene_asyn"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed 1 --n_training_threads 1 --n_rollout_threads 1 --use_eval --render_episodes 5 \
      --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' \
      --model_dir "/home/tsing90/yxy_project/onpolicy/onpolicy/scripts/results/GridWorld/MiniGrid-MultiExploration-v0/rmappo/single_merge_local_new_rand_loc_step150_envs30_size31_ppo3_lr5e-4_grid_scene_asyn/wandb/run-20210716_165942-3e1w8ct0/files/" --ifi 0.5 --use_wandb --use_render --max_steps 150 --save_gifs --use_merge --grid_size 31  --use_local --use_random_pos --agent_view_size 7 --use_single
      
done