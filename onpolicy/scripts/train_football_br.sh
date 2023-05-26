#!/bin/sh
# exp param
env="Football"
# scenario="academy_pass_and_shoot_with_keeper"
# scenario="academy_run_pass_and_shoot_with_keeper"
scenario="academy_3_vs_1_with_keeper"
algo="mappo"
exp="sacl_3v1_model3_blue@350M"
# exp="psro_3v1_model3_blue@100M_again"
seed=0


# football param
num_red=3
num_blue=1
num_agents=4
representation="simple115v2"
rewards="scoring,checkpoints"
# rewards="scoring"

# train param
num_env_steps=500000000
episode_length=200
data_chunk_length=10 # 10, 20

# log param
log_interval=5
save_interval=50
save_ckpt_interval=250

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=5 # 100

# tune param
n_rollout_threads=200 # 1000
ppo_epoch=10 # 5, 10, 15
num_mini_batch=2 # 2, 4

# red br
# training_mode="red_br"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063714-2p5oicm0/files/350M"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063445-md6txb0a/files"
# red_valuenorm_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063445-md6txb0a/files"

# blue br
training_mode="blue_br"
red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063714-2p5oicm0/files/350M"
blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063445-md6txb0a/files"
blue_valuenorm_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1_500M/wandb/run-20230522_063445-md6txb0a/files"

echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=3 python train/train_football_competitive.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue} --zero_sum \
--num_env_steps ${num_env_steps} --training_mode ${training_mode} \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--data_chunk_length ${data_chunk_length} \
--use_recurrent_policy \
--save_interval ${save_interval} --log_interval ${log_interval} \
--save_ckpt_interval ${save_ckpt_interval} \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "chenjy" --wandb_name "football" --rewards ${rewards} \
--red_model_dir ${red_model_dir} --blue_model_dir ${blue_model_dir} --blue_valuenorm_dir ${blue_valuenorm_dir} \
# --blue_model_dir ${blue_model_dir} --red_model_dir ${red_model_dir} --red_valuenorm_dir ${red_valuenorm_dir} \

# red br
# --blue_model_dir ${blue_model_dir} --red_model_dir ${red_model_dir} --red_valuenorm_dir ${red_valuenorm_dir} \

# blue br
# --red_model_dir ${red_model_dir} --blue_model_dir ${blue_model_dir} --blue_valuenorm_dir ${blue_valuenorm_dir} \