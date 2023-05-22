#!/bin/sh
# exp param
env="Football"
# scenario="academy_3_vs_1_with_keeper"
# scenario="academy_pass_and_shoot_with_keeper"
scenario="academy_run_pass_and_shoot_with_keeper"
algo="mappo"
exp="rps@50M_True"
seed=1


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
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185552-3q43bqb6/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185608-1zvrb2ns/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sacl_pass_shoot/wandb/run-20230512_185624-39n0215i/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064647-3ak4zyya/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064712-3umlsevp/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/sp_pass_shoot/wandb/run-20230512_064738-1giem989/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_pass_shoot/wandb/run-20230514_083422-37rrl3j0/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_pass_shoot/wandb/run-20230514_083546-29c0knsb/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/fsp_pass_shoot/wandb/run-20230514_083643-35zkjcbm/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_pass_shoot_population25M/wandb/run-20230513_154055-1wd3azff/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_pass_shoot_population25M/wandb/run-20230513_154223-18z7pe0v/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/psro_pass_shoot_population25M/wandb/run-20230513_154247-6psp787d/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/rnad_ps_entropy01_again/wandb/run-20230516_085430-6puzczhb/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/rnad_ps_entropy01_again/wandb/run-20230516_085554-2g9sqz44/files/50M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_pass_and_shoot_with_keeper/mappo/rnad_ps_entropy01_again/wandb/run-20230516_085611-10nuf2rh/files/50M"

# run_pass_shoot
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185715-17m6wngm/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185744-24nqw0wx/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185758-1vqo9k1w/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064756-1yjqksmx/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064807-271xypkd/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064813-1oie460z/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083712-1z2k6tgs/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083734-2gzgtdjz/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083751-1lpf2jgb/files"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230513_163345-3e4ytns6/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230513_163452-2wta4trk/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230517_081854-2dc5fed2/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_rps_model0_load80M/wandb/run-20230516_202651-1g3u2u9q/files/20M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_rps_model1_load90M/wandb/run-20230516_202723-86u4hzft/files/10M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_rps_model2_load90M/wandb/run-20230516_202756-34hz539l/files/10M"

# run pass shoot 50M
model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185715-17m6wngm/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185744-24nqw0wx/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185758-1vqo9k1w/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064756-1yjqksmx/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064807-271xypkd/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sp_run_pass_shoot/wandb/run-20230512_064813-1oie460z/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083712-1z2k6tgs/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083734-2gzgtdjz/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/fsp_run_pass_shoot/wandb/run-20230514_083751-1lpf2jgb/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230513_163345-3e4ytns6/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230513_163452-2wta4trk/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/psro_run_pass_shoot_population375/wandb/run-20230517_081854-2dc5fed2/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_run_pass_shoot_wotransfer/wandb/run-20230514_092117-1rjpu27t/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_run_pass_shoot_wotransfer/wandb/run-20230514_092140-1hyyvm5f/files/50M"
model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/rnad_run_pass_shoot_wotransfer/wandb/run-20230514_092154-3m3j027u/files/50M"

# 3v1
# model_dir_list="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190340-x9n2omu3/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190401-27ydowni/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190414-3prnpswd/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064846-45icn2r5/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064857-2grkzhbk/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sp_3v1/wandb/run-20230512_064902-2at9k5p7/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1/wandb/run-20230514_084047-1kkdh69t/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1/wandb/run-20230514_084105-27qs4y4n/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/fsp_3v1/wandb/run-20230514_084135-1u0c9m8k/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163727-1pcnb6hx/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163748-q5fwog1s/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163805-2p8shzws/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/rnad_3v1_entropy01/wandb/run-20230516_085243-aup94dxv/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/rnad_3v1_entropy01/wandb/run-20230516_085319-2y6qs8fq/files/100M"
# model_dir_list+=" /home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/rnad_3v1_entropy01/wandb/run-20230516_085338-3p7qfz3b/files/100M"

CUDA_VISIBLE_DEVICES=3 python eval/eval_football_cross_play.py \
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
