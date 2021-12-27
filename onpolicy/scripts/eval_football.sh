#!/bin/sh
# exp param
env="Football"
scenario="academy_single_goal_versus_lazy"
algo="rmappo"
exp="eval_test"

# football param
num_agents=11
representation="simple115v2"

n_rollout_threads=1
episode_length=1000

# eval param
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_single_goal_versus_lazy/shared/dense/rollout50-minibatch4-epoch10/seed1/files"
eval_episodes=100
n_eval_rollout_threads=100 # 100

# video param
# dump_frequency=1
# log_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/logs/n10_train2_eval1_seed1_run2"
# write_full_episode_dumps
# render
# save_gifs
# --dump_frequency ${dump_frequency} --log_dir ${log_dir} \
# --write_full_episode_dumps --render --save_gifs \

seed=0

CUDA_VISIBLE_DEVICES=5 python eval/eval_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--n_rollout_threads ${n_rollout_threads} --episode_length ${episode_length} \
--use_eval --model_dir ${model_dir} \
--eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "zelaix" --use_wandb
