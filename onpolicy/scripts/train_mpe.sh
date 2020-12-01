#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=2
num_agents=2
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 3 --num_env_steps 20000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --use_wandb --data_chunk_length 2
    echo "training is done!"
done
