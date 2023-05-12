#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
# scenario="academy_pass_and_shoot_with_keeper"
# scenario="academy_run_pass_and_shoot_with_keeper"
algo="mappo"
exp="3v1@50M"
seed=1


# football param
num_red=3
num_blue=1
num_agents=4
representation="simple115v2"
# rewards="scoring"
rewards="scoring,checkpoints"

# train param
num_env_steps=200000000
episode_length=200
data_chunk_length=10 # 10, 20

# log param
log_interval=20000
save_interval=20000

# eval param
eval_interval=400000
eval_episodes=500
n_eval_rollout_threads=100 # 100

# tune param
n_rollout_threads=100 # 1000
ppo_epoch=10 # 5, 10, 15
num_mini_batch=2 # 2, 4

# pass_shoot
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230511_120028-22s47f2m/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230511_120054-13jbwmtd/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230511_120125-2dqf3yn7/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot_scoring_checkpoint/wandb/run-20230509_162611-hcxi6g1u/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot_scoring_checkpoint/wandb/run-20230509_162640-rvfn0mxp/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot_scoring_checkpoint/wandb/run-20230509_162655-sslh60f4/files/60M"

# run_pass_shoot
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230511_120149-i3feoci8/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230511_120200-216n84gs/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230511_120219-19t0on68/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230510_161123-1r9yxdnk/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230510_161157-1wp0hx6b/files/60M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230510_161225-2rtofoe8/files/60M"

# 3v1
model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230511_120251-3caa7sk7/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230511_120319-1zocrnt8/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230511_120337-3aij7ga2/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230510_160142-2gbicx88/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230510_160230-2fgwz31p/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230510_160249-3gp37cos/files/50M"

CUDA_VISIBLE_DEVICES=2 python eval/eval_football_cross_play.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue} --zero_sum \
--num_env_steps ${num_env_steps} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--data_chunk_length ${data_chunk_length} \
--save_interval ${save_interval} --log_interval ${log_interval} \
--use_eval \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "chenjy" --wandb_name "football" --rewards ${rewards} \
--red_model_dir "${model_dir_list}" --blue_model_dir "${model_dir_list}" \
--use_recurrent_policy \
--use_wandb \
