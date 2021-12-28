#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="eval"
seed=1

# football param
num_agents=3
representation="simple115v2"

# set to 1 because train env is not used
n_rollout_threads=1 

# eval param
episode_length=200
eval_episodes=100
n_eval_rollout_threads=100
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_3_vs_1_with_keeper/shared/sparse/rollout50-minibatch4-epoch15/seed1/files"


echo "evaluate ${eval_episodes} episodes"

CUDA_VISIBLE_DEVICES=0 python eval/eval_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--n_rollout_threads ${n_rollout_threads} \
--use_eval \
--episode_length ${episode_length} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} --model_dir ${model_dir} \
--user_name "zelaix" --use_wandb
