#!/bin/sh
# exp param
env="Football"
# scenario="academy_3_vs_1_with_keeper"
# scenario="academy_pass_and_shoot_with_keeper"
scenario="academy_run_pass_and_shoot_with_keeper"
algo="mappo"
exp="rps_v4@300M_True"
seed=0


# football param
num_red=2
num_blue=1
num_agents=3
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
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_ps_500M/wandb/run-20230522_063900-2vyzs3ri/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_ps_500M/wandb/run-20230522_063930-jhbalacf/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_ps_500M/wandb/run-20230522_063942-23pehbwg/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_ps_500M/wandb/run-20230526_091349-gnzzrvao/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_ps_500M/wandb/run-20230526_091440-1x0cs0kd/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_ps_500M/wandb/run-20230526_091454-2drwrdg9/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_ps_500M/wandb/run-20230608_111946-1is43gn1/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_ps_500M/wandb/run-20230608_112122-252a8dv9/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_ps_500M/wandb/run-20230608_112140-hzh29vkc/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_ps_500M/wandb/run-20230522_062726-1kmsnyue/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_ps_500M/wandb/run-20230522_062951-2zp9yvbh/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_ps_500M/wandb/run-20230522_063111-2ok00htf/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/neurd_ps_500M/wandb/run-20230603_092913-3fpnxq0i/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/neurd_ps_500M/wandb/run-20230603_092940-n41i6gqe/files/300M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/neurd_ps_500M/wandb/run-20230603_093008-9nquujiy/files/300M"

# run_pass_shoot
model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_rps_v4_500M/wandb/run-20230619_075726-e9iasc3y/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_rps_v4_500M/wandb/run-20230619_075800-2yzgvkp9/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_rps_v4_500M/wandb/run-20230619_075815-2weaw0bn/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_rps_500M/wandb/run-20230522_095012-3rpmifp3/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_rps_500M/wandb/run-20230526_091634-1h71m5pa/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_rps_500M/wandb/run-20230526_091636-3m4x80c4/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_rps_500M/wandb/run-20230608_112335-2fjfca8y/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_rps_500M/wandb/run-20230608_112353-hldhplnl/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_rps_500M/wandb/run-20230608_112424-292s2q5m/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_rps_500M/wandb/run-20230603_092314-1w8qjkzg/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_rps_500M/wandb/run-20230603_092402-e00oqsr1/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_rps_500M/wandb/run-20230603_092433-2n140teq/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/neurd_rps_500M/wandb/run-20230603_093450-ukanuehp/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/neurd_rps_500M/wandb/run-20230603_093512-1if2bwxn/files/300M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/neurd_rps_500M/wandb/run-20230603_093540-39xuz0rg/files/300M"

# 3v1
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063445-md6txb0a/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063705-341vdo3t/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063714-2p5oicm0/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1_500M/wandb/run-20230525_045802-287bxf67/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1_500M/wandb/run-20230531_094908-ka7bolux/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1_500M/wandb/run-20230531_095041-pc7zb6uh/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1_500M/wandb/run-20230530_105950-3nnfk334/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1_500M/wandb/run-20230530_110027-5p52ko1y/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1_500M/wandb/run-20230530_110115-1m0by60l/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_500M/wandb/run-20230522_061755-30c75xeu/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_500M/wandb/run-20230522_061825-1rvjt5sr/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_500M/wandb/run-20230522_061843-3uj8ms0o/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/neurd_3v1_500M/wandb/run-20230603_092727-ql2puxcl/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/neurd_3v1_500M/wandb/run-20230603_092804-13umhm80/files/400M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/neurd_3v1_500M/wandb/run-20230603_092831-2ive4suq/files/400M"

CUDA_VISIBLE_DEVICES=0 python eval/eval_football_cross_play.py \
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
