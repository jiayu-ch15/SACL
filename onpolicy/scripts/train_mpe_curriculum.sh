#!/bin/sh
# exp config
exp="2_corner-sacl-rb_variance**5-staleness_20-wo_time"
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
valuenorm_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230404_053322-3a5vx194/files"
num_env_steps=50000000
episode_length=200
n_rollout_threads=100
ppo_epoch=5
log_interval=5
save_interval=50
# curriculum config
prob_curriculum=0.7
curriculum_buffer_size=10000
update_method="fps"
sample_metric="variance"
max_staleness=20
# user name
user_name="zelaix"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=6 python train/train_mpe_curriculum.py \
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
--max_staleness ${max_staleness} \
--user_name "zelaix" \
--wandb_name ${wandb_name}
