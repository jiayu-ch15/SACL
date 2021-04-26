#!/bin/sh
env="GridWorld"
scenario="MiniGrid-Human-v0"
num_agents=2
num_preies=2
num_obstacles=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py --use_human_command --direction_alpha 0.5 \
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_preies ${num_preies} --num_obstacles ${num_obstacles} \
      --seed 2 --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 1 \
      --cnn_layers_params '32,3,1,1' --model_dir "xxx" --ifi 0.5
done
