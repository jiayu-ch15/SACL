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
rewards="scoring,checkpoints"
# rewards="scoring"

# train param
num_env_steps=200000000
episode_length=200
data_chunk_length=10 # 10, 20

# log param
log_interval=20000
save_interval=20000
save_ckpt_interval=5000000

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=5 # 100

# tune param
n_rollout_threads=5 # 1000
ppo_epoch=5 # 5, 10, 15
num_mini_batch=2 # 2, 4

# CL
prob_curriculum=0.7
curriculum_buffer_size=10000
beta=1.0
alpha=0.7
num_critic=1
update_method="fps"
sample_metric="rb_variance"


echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=1 python train/train_football_curriculum.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue} --zero_sum \
--num_env_steps ${num_env_steps} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--data_chunk_length ${data_chunk_length} \
--save_interval ${save_interval} --log_interval ${log_interval} \
--use_eval --save_ckpt_interval ${save_ckpt_interval} \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "chenjy" --wandb_name "football" --rewards ${rewards} \
--prob_curriculum ${prob_curriculum} --curriculum_buffer_size ${curriculum_buffer_size} \
--beta ${beta} --alpha ${alpha} --num_critic ${num_critic} \
--sample_metric ${sample_metric} --update_method ${update_method} \
--use_wandb \
