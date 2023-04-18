#!/bin/bash
# exp config
exp="metric@75M"
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


model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_131954-2937oo8o/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_131959-3rstbiw6/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/uniform/wandb/run-20230410_132007-1tfhu8gn/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_unif/wandb/run-20230411_083102-1xor7hxa/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_unif/wandb/run-20230411_083248-27m7pbv5/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_unif/wandb/run-20230411_083256-3jvxuusf/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132308-1zt784gi/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132346-2rf96j6u/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132353-3uxxrma8/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061330-2vmcdj33/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061427-35ylucuj/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061439-23yn83a0/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/TDerror/wandb/run-20230414_031028-2lhco2rk/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/TDerror/wandb/run-20230414_031055-31kdxlwm/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/TDerror/wandb/run-20230414_031108-n4q2a04n/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031641-n9dn372p/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031654-1r8njnjd/files/75M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031706-2kot9l67/files/75M"

# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132308-1zt784gi/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132346-2rf96j6u/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132353-3uxxrma8/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_bias/wandb/run-20230411_084722-1yq61vnp/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_bias/wandb/run-20230411_084818-1t6izewa/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_bias/wandb/run-20230411_084826-pakikq7c/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_1var_1bias/wandb/run-20230411_084854-28hpmltl/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_1var_1bias/wandb/run-20230411_084907-2xwxuj8y/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_1var_1bias/wandb/run-20230411_084920-2aihduhb/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_03bias/wandb/run-20230414_031520-1au56hm5/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_03bias/wandb/run-20230414_031539-20k7w2mu/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_03bias/wandb/run-20230414_031600-1ahueu1k/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_05bias/wandb/run-20230414_031421-2jmxganc/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_05bias/wandb/run-20230414_031435-3hra1xz4/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_05bias/wandb/run-20230414_031454-2xf83xvw/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031641-n9dn372p/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031654-1r8njnjd/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031706-2kot9l67/files/100M"

# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132308-1zt784gi/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132346-2rf96j6u/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble_individual_variance/wandb/run-20230410_132353-3uxxrma8/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061330-2vmcdj33/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061427-35ylucuj/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/rb_var/wandb/run-20230412_061439-23yn83a0/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble5/wandb/run-20230414_031822-5u08ul6u/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble5/wandb/run-20230414_031834-2ecyyeiz/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/ensemble5/wandb/run-20230414_031857-31e7seum/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/adv_var_ensemble4/wandb/run-20230414_132555-1z2mzjtf/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/adv_var_ensemble4/wandb/run-20230414_132602-266vsp6v/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/adv_var_ensemble4/wandb/run-20230414_132617-13dffzu0/files/100M"


# user name
user_name="chenjy"
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
