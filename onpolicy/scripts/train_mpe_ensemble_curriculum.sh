#!/bin/sh
# exp config
exp="rb_var"
algo="mappo"
seed=0
# env config
env="MPE"
scenario="simple_tag_corner"
horizon=200
corner_min=1.0
corner_max=2.0
num_adv=3
num_good=1
num_landmarks=2
# policy config
attn_size=32
# training config
training_mode="self_play"
num_env_steps=100000000
episode_length=200
n_rollout_threads=100
ppo_epoch=5
log_interval=5
save_interval=50
save_ckpt_interval=250
# curriculum config
prob_curriculum=0.7
curriculum_buffer_size=10000
beta=1.0
alpha=0.0
num_critic=3
update_method="fps"
sample_metric="rb_variance"
# user name
user_name="chenjy"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train/train_mpe_ensemble_curriculum.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--prob_curriculum ${prob_curriculum} \
--curriculum_buffer_size ${curriculum_buffer_size} \
--update_method ${update_method} --sample_metric ${sample_metric} \
--alpha ${alpha} --beta ${beta} --num_critic ${num_critic} \
--max_staleness ${max_staleness} \
--hard_boundary \
--user_name ${user_name} \
--wandb_name ${wandb_name} \
