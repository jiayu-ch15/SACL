#!/bin/sh
# exp params
env="Football"
# scenario="academy_3_vs_1_with_keeper"
# scenario="academy_pass_and_shoot_with_keeper"
scenario="academy_run_pass_and_shoot_with_keeper"
algo="mappo"
exp="render"
seed=0

# football params
num_agents=3
num_red=2
num_blue=1
representation="simple115v2"
dump_frequency=1

# sp
training_mode='self_play'

# render params
render_episodes=20
n_rollout_threads=1

red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185715-17m6wngm/files/100M"
blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_run_pass_and_shoot_with_keeper/mappo/sacl_run_pass_shoot/wandb/run-20230512_185715-17m6wngm/files/100M"

# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190340-x9n2omu3/files/100M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190340-x9n2omu3/files/100M"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190414-3prnpswd/files/100M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/sacl_3v1/wandb/run-20230512_190414-3prnpswd/files/100M"

# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163727-1pcnb6hx/files/100M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163727-1pcnb6hx/files/100M"
# red_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163805-2p8shzws/files/100M"
# blue_model_dir="/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/Football/academy_3_vs_1_with_keeper/mappo/psro_3v1_population375/wandb/run-20230513_163805-2p8shzws/files/100M"

# --save_videos is preferred instead of --save_gifs 
# because .avi file is much smaller than .gif file

echo "render ${render_episodes} episodes"

CUDA_VISIBLE_DEVICES=0 python render/render_football.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_red ${num_red} --num_blue ${num_blue}  \
--use_render \
--render_episodes ${render_episodes} --n_rollout_threads ${n_rollout_threads} \
--red_model_dir ${red_model_dir} --blue_model_dir ${blue_model_dir} \
--red_valuenorm_dir ${red_model_dir} --blue_valuenorm_dir ${blue_model_dir} \
--user_name "chenjy" \
--use_wandb \
--use_recurrent_policy \
--training_mode ${training_mode} \
--save_videos \