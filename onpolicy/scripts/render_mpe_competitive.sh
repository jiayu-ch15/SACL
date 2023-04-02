#!/bin/sh
# exp config
exp="render_test"
algo="mappo"
seed=0
# env config
env="MPE"
scenario="simple_tag_corner"
horizon=200
corner_min=-1.0
corner_max=1.0
num_adv=3
num_good=1
num_landmarks=2
# policy config
attn_size=32
# render config
render_episodes=5
n_rollout_threads=1
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-1_max1_sp_test/wandb/run-20230401_165058-ln836piu/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min2_max3_sp_test/wandb/run-20230401_165535-2ynuub3j/files"
# user name
user_name="zelaix"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=7 xvfb-run -s "-screen 0 1400x900x24" python render/render_mpe_competitive.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--use_render --save_gifs --render_episodes ${render_episodes} \
--n_rollout_threads ${n_rollout_threads} --model_dir ${model_dir}
