#!/bin/sh
env="HideAndSeek"
scenario_name="quadrant"
num_seekers=1
num_hiders=1
algo="test"
seed_max=1

echo "env is ${env}, scenario is ${scenario_name}, num_seekers is ${num_seekers}, num_hiders is ${num_hiders}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_hns.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_seekers ${num_seekers} --num_hiders ${num_hiders} --seed ${seed} --n_rollout_threads 1 --num_mini_batch 1 --episode_length 10 --num_env_steps 5000000 --ppo_epoch 5
    echo "training is done!"
done
