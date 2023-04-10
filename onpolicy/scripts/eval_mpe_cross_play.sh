#!/bin/bash
# exp config
exp="eval-1h_unif-xp@25M"
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
# train config
n_rollout_threads=1
# eval config
n_eval_rollout_threads=100
eval_episodes=100


model_dir_list="/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sp/wandb/run-20230407_150411-34nhh0l6/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sp/wandb/run-20230408_140127-1yj7635h/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sp/wandb/run-20230408_140156-4uclnrvv/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-unif-wo_time-test/wandb/run-20230407_152534-qs9mkfyl/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-unif-wo_time/wandb/run-20230408_140427-3hjjd3ny/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-unif-wo_time/wandb/run-20230408_140446-pb0zt62e/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-rb_var-wo_time-test/wandb/run-20230407_152603-14fxed1f/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-rb_var-wo_time/wandb/run-20230408_140247-2iv6bw7m/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_unif-sacl-rb_var-wo_time/wandb/run-20230408_140322-2cwbozh5/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-unif-wo_time/wandb/run-20230408_140744-28glal3n/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-unif-wo_time/wandb/run-20230408_140751-e6kh2l19/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-unif-wo_time/wandb/run-20230408_140819-25uqax9t/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-rb_var-wo_time/wandb/run-20230408_140906-vib1w5uw/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-rb_var-wo_time/wandb/run-20230408_140917-27j07hue/files/25M"
model_dir_list+=" /home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1h_crnr-sacl-rb_var-wo_time/wandb/run-20230408_140927-1k27xt7a/files/25M"

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
user_name="zelaix"
wandb_name="sacl"


CUDA_VISIBLE_DEVICES=7 python eval/eval_mpe_cross_play.py \
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
