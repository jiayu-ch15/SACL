#!/bin/sh
env="SMARTS"
scenario="vulner" #simple_speaker_listner   simple_spread
num_agents=1
algo="rmappo"
exp="test"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python -W ignore train/train_smarts.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 1000 --num_env_steps 20000000 --ppo_epoch 15 --gain 0.01 --lr 1e-4 --use_wandb
    echo "training is done!"
done
