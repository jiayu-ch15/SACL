#!/bin/bash
# exp config
exp="eval-corner_unif-xp@50M"
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
# train config
n_rollout_threads=1
# eval config
n_eval_rollout_threads=100
eval_episodes=100


model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_131954-2937oo8o/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_131959-3rstbiw6/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_132007-1tfhu8gn/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_mean_variance/wandb/run-20230410_132220-1etry7dv/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_mean_variance/wandb/run-20230410_132228-3coaw48c/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_mean_variance/wandb/run-20230410_132233-1qqrzsdj/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132308-1zt784gi/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132346-2rf96j6u/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132353-3uxxrma8/files/50M"

# model_dir_list="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072750-3c2mvcmr/files/25M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072904-24wskyj8/files/25M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072931-3vs4ydnw/files/25M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072750-3c2mvcmr/files/50M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072904-24wskyj8/files/50M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072931-3vs4ydnw/files/50M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072750-3c2mvcmr/files/75M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072904-24wskyj8/files/75M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072931-3vs4ydnw/files/75M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072750-3c2mvcmr/files/100M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072904-24wskyj8/files/100M"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230406_072931-3vs4ydnw/files/100M"

# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform-wo_time/wandb/run-20230406_073247-1wgu8u79/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform-wo_time/wandb/run-20230406_073609-485r8aa5/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform-wo_time/wandb/run-20230406_073656-31wh2vdy/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance-wo_time/wandb/run-20230406_073843-u9p7a6fi/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance-wo_time/wandb/run-20230406_073910-y10rg3oc/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance-wo_time/wandb/run-20230406_073937-36gtgpih/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_3-wo_time/wandb/run-20230406_074951-1vqxsu5c/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_3-wo_time/wandb/run-20230406_075121-16rjcfet/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_3-wo_time/wandb/run-20230406_075142-1m6ecsc4/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_2-wo_time/wandb/run-20230406_075411-2r3khuhz/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_2-wo_time/wandb/run-20230406_075647-3loxmu3n/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_2-wo_time/wandb/run-20230406_075720-37k3lxej/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_mean-wo_time/wandb/run-20230406_134657-36er7jls/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_mean-wo_time/wandb/run-20230406_134922-2y6fq9ez/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance_mean-wo_time/wandb/run-20230406_135016-ump76aeo/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform/wandb/run-20230406_082504-3vzjimk6/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform/wandb/run-20230406_082741-2tu0op5n/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform/wandb/run-20230406_082933-utgphda5/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance/wandb/run-20230406_083117-yp8par7v/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance/wandb/run-20230406_083227-11xb5r2h/files"
# model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance/wandb/run-20230406_083416-1u6wms83/files"
# model_dir_list+=" "
# user name
user_name="chenjy"
wandb_name="sacl"


CUDA_VISIBLE_DEVICES=0 python eval/eval_mpe_cross_play.py \
--experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
--env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
--corner_min ${corner_min} --corner_max ${corner_max} --hard_boundary \
--num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
--use_attn --attn_size ${attn_size} --use_recurrent_policy \
--n_rollout_threads ${n_rollout_threads} \
--use_eval --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--red_model_dir "${model_dir_list}" --blue_model_dir "${model_dir_list}" \
--use_wandb
