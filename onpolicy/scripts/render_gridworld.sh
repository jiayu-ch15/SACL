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
      --direction_alpha 0 --max_steps 60 --use_fixed_goal_pos \
      --seed 2 --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 1 \
      --cnn_layers_params '32,3,1,1' --save_gifs \
      --model_dir "/home/huangruixin/render/GridWorld/dirrew_2agent_0obstacle_alpha0.005" --ifi 0.5 --use_wandb
done

