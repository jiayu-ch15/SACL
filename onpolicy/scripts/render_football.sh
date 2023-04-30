#!/bin/sh
# exp params
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="render"
seed=1

# football params
num_agents=3
representation="simple115v2"
dump_frequency=1

# render params
render_episodes=3
n_rollout_threads=1
model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/rmappo/3v1_sp/wandb/run-20230426_164034-besg41d5/files"

# --save_videos is preferred instead of --save_gifs 
# because .avi file is much smaller than .gif file

echo "render ${render_episodes} episodes"

CUDA_VISIBLE_DEVICES=0 python render/render_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--use_render \
--render_episodes ${render_episodes} --n_rollout_threads ${n_rollout_threads} \
--model_dir ${model_dir} \
--save_videos \
--user_name "chenjy" \
--use_wandb \
