#!/bin/sh
# exp param
env="Football"
# scenario="academy_pass_and_shoot_with_keeper"
# scenario="academy_run_pass_and_shoot_with_keeper"
scenario="academy_3_vs_1_with_keeper"
algo="mappo"
exp="sacl_3v1"
seed=2


# football param
num_red=3
num_blue=1
num_agents=4
representation="simple115v2"
rewards="scoring,checkpoints"
# rewards="scoring"

# train param
num_env_steps=500000000
episode_length=200
data_chunk_length=10 # 10, 20

# log param
log_interval=5
save_interval=50
save_ckpt_interval=250

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=5 # 100

# tune param
n_rollout_threads=200 # 1000
ppo_epoch=10 # 5, 10, 15
num_mini_batch=2 # 2, 4

# CL
prob_curriculum=0.7
curriculum_buffer_size=10000
beta=1.0
alpha=0.0
num_critic=3
update_method="fps"
sample_metric="ensemble_var_add_bias"

# sp
training_mode='self_play'


echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=7 python train/train_football_curriculum.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue} --zero_sum \
--num_env_steps ${num_env_steps} --training_mode ${training_mode} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--data_chunk_length ${data_chunk_length} \
--use_recurrent_policy \
--save_interval ${save_interval} --log_interval ${log_interval} \
--save_ckpt_interval ${save_ckpt_interval} \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "chenjy" --wandb_name "football" --rewards ${rewards} \
--prob_curriculum ${prob_curriculum} --curriculum_buffer_size ${curriculum_buffer_size} \
--beta ${beta} --alpha ${alpha} --num_critic ${num_critic} \
--sample_metric ${sample_metric} --update_method ${update_method} \
