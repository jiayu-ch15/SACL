#!/bin/sh
env="GridWorld"
scenario="MiniGrid-DoorKey-5x5-v0"
num_agents=1
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 python render/render_gridworld.py --save_gifs --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 5 --cnn_layers_params '32,3,1,1' --model_dir "xxx" --ifi 0.5
done
