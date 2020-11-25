#!/bin/sh
env="StarCraft2"
map="corridor"
algo="rmappo"
exp="conv1d_decentralizedv_T_coef2_lr1e-4_entropycoef0.005_norm0.5"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 50000000 --ppo_epoch 15 --eval --use_single_network --use_value_active_masks --use_centralized_V --hidden_size 256 --value_loss_coef 2 --use_conv1d --lr 1e-4 --entropy_coef 0.005 --max_grad_norm 0.5
    echo "training is done!"
done