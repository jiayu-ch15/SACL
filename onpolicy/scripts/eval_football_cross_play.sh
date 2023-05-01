#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="debug"
seed=1


# football param
num_red=3
num_blue=1
num_agents=4
representation="simple115v2"
# rewards="scoring"
rewards="scoring"

# train param
num_env_steps=200000000
episode_length=200
data_chunk_length=10 # 10, 20

# log param
log_interval=20000
save_interval=20000

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=100 # 100

# tune param
n_rollout_threads=100 # 1000
ppo_epoch=5 # 5, 10, 15
num_mini_batch=2 # 2, 4

model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/rmappo/3v1_sp_rewards_scoring/wandb/run-20230430_151932-3usnwd94/files"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/rmappo/3v1_sp_rewards_scoring_checkpoints/wandb/run-20230430_151915-p1lbpo8v/files"

CUDA_VISIBLE_DEVICES=1 python eval/eval_football_cross_play.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue} --zero_sum \
--num_env_steps ${num_env_steps} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--data_chunk_length ${data_chunk_length} \
--save_interval ${save_interval} --log_interval ${log_interval} \
--use_eval \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "chenjy" --wandb_name "football" --rewards ${rewards} \
--red_model_dir "${model_dir_list}" --blue_model_dir "${model_dir_list}" \
--use_wandb \
