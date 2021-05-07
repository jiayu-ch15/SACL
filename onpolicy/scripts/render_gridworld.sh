#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed 2 --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 1 \
      --cnn_layers_params '32,3,1,1' --model_dir "/home/yuchao/project/onpolicy/onpolicy/scripts/results/GridWorld/MiniGrid-MultiExploration-v0/rmappo/minigrid_world_res/run20/models" --ifi 0.5 --use_wandb --visualize_input --use_merge
done

