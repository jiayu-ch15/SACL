#!/bin/sh
# exp config
exp="render"
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
# render config
render_episodes=5
n_rollout_threads=1
red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/fsp/wandb/run-20230418_130715-1nvu4lrq/files/40M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/fsp/wandb/run-20230418_130715-1nvu4lrq/files/40M"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_corner/wandb/run-20230411_083339-rdx6xz91/files/40M"
blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_corner/wandb/run-20230411_083339-rdx6xz91/files/40M"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031641-n9dn372p/files/40M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031641-n9dn372p/files/40M"

# user name
user_name="chenjy"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 xvfb-run -s "-screen 0 1400x900x24" python render/render_mpe_competitive.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} --hard_boundary \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--use_render --save_gifs --render_episodes ${render_episodes} \
--n_rollout_threads ${n_rollout_threads} \
--red_model_dir ${red_model_dir} --blue_model_dir ${blue_model_dir} \
--red_valuenorm_dir ${red_model_dir} --blue_valuenorm_dir ${blue_model_dir} \
--use_wandb
