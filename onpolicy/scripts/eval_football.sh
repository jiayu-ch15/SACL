#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="eval_test"

# football param
num_agents=3
representation="simple115v2"

n_rollout_threads=2

# eval param
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_3_vs_1_with_keeper/rmappo/shared/rollout10-minibatch4-epoch15/seed1/files"
eval_episodes=20
n_eval_rollout_threads=1 # 100

seed=1

CUDA_VISIBLE_DEVICES=0 python eval/eval_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--n_rollout_threads ${n_rollout_threads} \
--use_eval --model_dir ${model_dir} \
--eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "zelaix" --use_wandb
