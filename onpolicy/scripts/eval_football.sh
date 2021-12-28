#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="eval_test"
seed=1

# football param
num_agents=3
representation="simple115v2"

# episode param
n_rollout_threads=1 # set to 1 because train env is not used
episode_length=200

# eval param
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_3_vs_1_with_keeper/shared/sparse/rollout50-minibatch4-epoch15/seed1/files"
eval_episodes=100
n_eval_rollout_threads=100 # 100


echo "evaluate ${eval_episodes} episodes"

CUDA_VISIBLE_DEVICES=7 python eval/eval_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--n_rollout_threads ${n_rollout_threads} --episode_length ${episode_length} \
--use_eval --model_dir ${model_dir} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "zelaix" --use_wandb
