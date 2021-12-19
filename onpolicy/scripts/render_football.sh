#!/bin/sh
# exp params
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="render"

# football params
num_agents=3
representation="simple115v2"
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_3_vs_1_with_keeper/rmappo/shared/rollout10-minibatch4-epoch15/seed3/files"
log_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/football/academy_3_vs_1_with_keeper/rmappo/shared/rollout10-minibatch4-epoch15/seed3/videos"
dump_frequency=1

# render params
episode_length=200
n_rollout_threads=1
render_episodes=100
num_env_steps=200


CUDA_VISIBLE_DEVICES=0 python render/render_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed 3 \
--num_agents ${num_agents} --representation ${representation} \
--use_render \
--model_dir ${model_dir} --log_dir ${log_dir} --dump_frequency ${dump_frequency} \
--num_env_steps ${num_env_steps} --render_episodes ${render_episodes} \
--n_rollout_threads ${n_rollout_threads} \
--episode_length ${episode_length} \
--user_name "zelaix" --wandb_name "zelaix"
