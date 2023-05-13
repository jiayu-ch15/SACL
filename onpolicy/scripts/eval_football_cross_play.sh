#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
# scenario="academy_pass_and_shoot_with_keeper"
# scenario="academy_run_pass_and_shoot_with_keeper"
algo="mappo"
exp="3v1_debug@190M"
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

model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/150M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/150M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/150M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/180M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/180M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/180M"

# pass_shoot
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185552-3q43bqb6/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185608-1zvrb2ns/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185624-39n0215i/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064647-3ak4zyya/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064712-3umlsevp/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064738-1giem989/files/50M"

# run_pass_shoot
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185715-17m6wngm/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185744-24nqw0wx/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185758-1vqo9k1w/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064756-1yjqksmx/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064807-271xypkd/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064813-1oie460z/files/50M"

# 3v1
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190340-x9n2omu3/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190401-27ydowni/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190414-3prnpswd/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064857-2grkzhbk/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064902-2at9k5p7/files/50M"

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