#!/bin/sh
env="StarCraft2"
map="3s5z_vs_3s6z"
algo="rmappo"
exp="removecenterxy"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 50000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_alive_state --add_move_state --add_distance_state --add_enemy_action_state --add_agent_id --add_xy_state --user_name "zoeyuchao"
    echo "training is done!"
done