#!/bin/bash
# exp config
exp="debug@40M"
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

# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031641-n9dn372p/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031654-1r8njnjd/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/1var_07bias/wandb/run-20230414_031706-2kot9l67/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_corner/wandb/run-20230411_083339-rdx6xz91/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_corner/wandb/run-20230411_083406-a4x0xfwx/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/sp_corner/wandb/run-20230411_083423-2qq3rk4f/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/fsp/wandb/run-20230418_130715-1nvu4lrq/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/fsp/wandb/run-20230418_130729-39b37izy/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/fsp/wandb/run-20230418_130747-2f1bcs9s/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/psro/wandb/run-20230421_131147-3vcovmuo/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/psro/wandb/run-20230421_131156-a9kwyf9x/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/psro/wandb/run-20230421_131208-2pw62a01/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_rnad/wandb/run-20230509_134231-2kdswwzg/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_rnad/wandb/run-20230509_134240-tqvjucxx/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/corner_rnad/wandb/run-20230509_134248-2vk7pv32/files/40M"

# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sacl/wandb/run-20230507_103514-1nl9q1sd/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sacl/wandb/run-20230507_103523-2incs51i/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sacl/wandb/run-20230507_103531-2lr1ikex/files/40M"
model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sp/wandb/run-20230411_083102-1xor7hxa/files/40M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sp/wandb/run-20230411_083248-27m7pbv5/files/40M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_sp/wandb/run-20230411_083256-3jvxuusf/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_fsp/wandb/run-20230507_103846-3g678fph/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_fsp/wandb/run-20230507_103857-1erz3k4p/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_fsp/wandb/run-20230507_103905-lyv7zkzm/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_psro/wandb/run-20230507_103940-sobiz3b3/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_psro/wandb/run-20230507_103951-1xdr8ex2/files/40M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_psro/wandb/run-20230507_104001-322zyf8q/files/40M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_rnad/wandb/run-20230512_164342-1t517bqu/files/40M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_rnad/wandb/run-20230512_164346-2oj4v252/files/40M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/full_rnad/wandb/run-20230512_164356-1xuvtad9/files/40M"

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


CUDA_VISIBLE_DEVICES=3 python eval/eval_mpe_cross_play.py \
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
