#!/bin/sh
# exp config
exp="render"
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
# render config
render_episodes=5
n_rollout_threads=1
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-1_max1_sp/wandb/run-20230402_101310-37nr8qxp/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-2_max2_sp/wandb/run-20230402_093218-3aqr22uz/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-3_max3_sp/wandb/run-20230402_092732-2klutofi/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-3_max3-sp/wandb/run-20230402_172744-2jsob8uj/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min2_max3-sacl-random_sample-w_landmark-wo_time/wandb/run-20230402_171801-wqrg0zo6/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min2_max3-sacl-random_sample-wo_landmark-wo_time/wandb/run-20230402_172024-umdgscrs/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min-2_max2-sp/wandb/run-20230402_172524-vs56mms4/files"
# model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min1_max2-sacl-random_sample-w_landmark-wo_time/wandb/run-20230402_171212-2hxcxq6b/files"
model_dir="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/min1_max2-sacl-random_sample-wo_landmark-wo_time/wandb/run-20230402_172220-2tkd9hne/files"
# model_dir=""


# user name
user_name="zelaix"
wandb_name="sacl"


echo "exp is ${exp}, env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=5 xvfb-run -s "-screen 0 1400x900x24" python render/render_mpe_competitive.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--use_render --save_gifs --render_episodes ${render_episodes} \
--n_rollout_threads ${n_rollout_threads} \
--red_model_dir ${model_dir} --blue_model_dir ${model_dir} \
--use_wandb
