#!/bin/sh
env="Highway"
scenario="highway-v0"
task="defend"
n_defenders=1
n_attackers=0
n_dummies=0
algo="rmappo"
exp="debug"
seed_max=1
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario}, task is ${task}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 1 --horizon 40 --episode_length 100 --log_interval 1 --use_render --use_wandb
    echo "training is done!"
done
