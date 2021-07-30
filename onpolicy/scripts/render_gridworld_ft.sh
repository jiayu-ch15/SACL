#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="ft_rrt"
exp="debug "
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}"
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld_ft.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_eval --render_episodes 1 \
      --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' \
      --ifi 0.3 --use_wandb --max_steps 50 --use_merge --grid_size 31 --use_local --use_random_pos --agent_view_size 7 --use_single --user_name "gaojiaxuan" \
      --use_render --save_gifs 
done 