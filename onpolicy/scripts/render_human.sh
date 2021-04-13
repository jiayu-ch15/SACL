#!/bin/sh
env="Human"
scenario="simple_tag"
num_landmarks=2
num_good_agents=1
num_adversaries=3
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_human.py --use_fixed_prey --save_gifs --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --model_dir "/home/yuchao/project/onpolicy/onpolicy/scripts/results/Human/simple_tag/rmappo/debug/run19/models/"
done
