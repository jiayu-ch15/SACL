#!/bin/sh
env="single_navigation"
algo="test"
step_unknown=2

echo "env is ${env}"

CUDA_VISIBLE_DEVICES=6 python eval_MPE.py --env_name ${env} --algorithm_name ${algo} --recurrent_policy --num_agents 1 --num_landmarks 3 --episode_length 60 --step_unknown ${step_unknown} --model_dir "/home/yuchao/project/mappo-ssd/results/single_navigation/lstm"+${step_unknown}"/run1/models" --eval_episodes 100
