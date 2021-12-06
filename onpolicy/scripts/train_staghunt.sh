#!/bin/sh
env="StagHuntGW"
num_agents=2
algo="rmappo"
exp="separate"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_staghunt.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --user_name "zelaix" \
    --num_agents ${num_agents} --seed 1 --n_training_threads 1 \
    --n_rollout_threads 256 --num_mini_batch 1 --episode_length 50 \
    --num_env_steps 7000000 --ppo_epoch 4 --gain 0.01 --lr 1e-3 \
    --critic_lr 1e-3 --wandb_name "zelaix" --share_policy
done
