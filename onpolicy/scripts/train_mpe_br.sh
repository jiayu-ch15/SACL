#!/bin/sh
# exp config
exp="matrpo_sacl_again_model3@1800K"
# exp="full_fsp_model3@35M-from_pretrained40M"
# exp="debug"
algo="mappo"
oppenent_name="matrpo"
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
training_mode="red_br"
blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/matrpo/matrpo_sacl_attn_again/wandb/run-20231107_122358-3kvs5r8k/files/1800K"
red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl/files/run-20230414_031641-n9dn372p/files"
red_valuenorm_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl/files/run-20230414_031641-n9dn372p/files"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl/files/run-20230414_031654-1r8njnjd/files/40M"
# red_valuenorm_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_sacl/files/run-20230414_031654-1r8njnjd/files/40M"
num_env_steps=200000000
episode_length=200
n_rollout_threads=100
ppo_epoch=5
log_interval=5
save_interval=50
save_ckpt_interval=250
warm_up=0 # 5M
# user name
user_name="chenjy"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python train/train_mpe_competitive.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--training_mode ${training_mode} \
--num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--save_ckpt_interval ${save_ckpt_interval} \
--hard_boundary \
--user_name ${user_name} \
--wandb_name ${wandb_name} \
--warm_up ${warm_up} \
--oppenent_name ${oppenent_name} \
--blue_model_dir ${blue_model_dir} \
--red_model_dir ${red_model_dir} --red_valuenorm_dir ${red_valuenorm_dir} \
# --use_wandb \
