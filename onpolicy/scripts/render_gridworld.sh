#!/bin/sh
env="GridWorld"
scenario="MiniGrid-Human-v0"
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed 2 --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 1 --cnn_layers_params '32,3,1,1' --model_dir "xxx" --ifi 0.5
done
