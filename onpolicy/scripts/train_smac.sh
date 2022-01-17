#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="rmappo"
exp="clean_state_xy_rollout16"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed 3 \
    --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 1 \
    --episode_length 400 --num_env_steps 25000000 --ppo_epoch 10 \
    --use_value_active_masks --use_eval --user_name "zoeyuchao"
done
