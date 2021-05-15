#!/bin/sh
env="StarCraft2"
map="3s_vs_5z"
algo="mappo"
exp="debug"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 \
    --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 \
    --use_value_active_masks  --user_name "zoeyuchao" --use_wandb --use_global_local_state --use_state_agent --use_recurrent_policy --use_stacked_frames --stacked_frames 4 #--use_eval
done
