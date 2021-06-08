#!/bin/sh
env="MVE"
scenario="3p1t2f"  # simple_speaker_listener # simple_spread
num_landmarks=2
num_agents=3
algo="mappo"
exp="MAPPO"
seed_max=10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mve.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --user_name "zoeyuchao" --num_agents ${num_agents} --num_landmarks ${num_landmarks} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 \
    --episode_length 25 --num_env_steps 5000000 --ppo_epoch 10 --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4 --wandb_name "tartrl" --use_recurrent_policy \
    --direction_alpha 0.1 --use_wandb
done
