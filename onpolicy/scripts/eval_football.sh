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
