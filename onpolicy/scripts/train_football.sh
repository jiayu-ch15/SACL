#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="shared-dense"
seed=1

# football param
num_agents=3
representation="simple115v2"
rewards="scoring,checkpoints"

# train param
num_env_steps=15000000
episode_length=200
n_rollout_threads=50 # 1000
num_mini_batch=4 # 2, 4
ppo_epoch=15 # 5, 10, 15
data_chunk_length=10 # 10, 20

# log param
log_interval=200000
save_interval=200000

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=50 # 100


echo "exp: ${exp} \t n_rollout_threads: ${n_rollout_threads} \t num_mini_batch: ${num_mini_batch} \t ppo_epoch: ${ppo_epoch}"

CUDA_VISIBLE_DEVICES=0 python train/train_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--rewards ${rewards} \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--num_mini_batch ${num_mini_batch} --data_chunk_length ${data_chunk_length} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--use_eval \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "zelaix" --wandb_name "football"
