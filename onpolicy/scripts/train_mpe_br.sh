#!/bin/sh
# exp config
exp="2_uniform-adv_br"
algo="mappo"
seed=0
# env config
env="MPE"
scenario="simple_tag_corner"
horizon=200
corner_min=-2.0
corner_max=2.0
num_adv=3
num_good=1
num_landmarks=2
# policy config
attn_size=32
# training config
training_mode="red_br"
blue_model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-2_max2_sp/wandb/run-20230402_093218-3aqr22uz/files"
# red_model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-2_max2_sp/wandb/run-20230402_093218-3aqr22uz/files"
num_env_steps=100000000
episode_length=200
n_rollout_threads=100
ppo_epoch=5
log_interval=5
save_interval=50

# user name
user_name="zelaix"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=7 python train/train_mpe_competitive.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--training_mode ${training_mode} --blue_model_dir ${blue_model_dir} \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--user_name "zelaix" \
--wandb_name ${wandb_name}
