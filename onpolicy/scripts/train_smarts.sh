#!/bin/sh
env="SMARTS"
scenario="straight"
num_agents=3
num_lanes=3
algo="rmappo"
exp="debug"
seed_max=1

echo "building scenario ${scenario} ..."
scl scenario build-attack --clean --num_agents ${num_agents} --num_lanes ${num_lanes} ../envs/smarts/SMARTS/scenarios/${scenario}
echo "build scenario ${scenario} successfully!"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python -W ignore train/train_smarts.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 10 --num_env_steps 2000000000 --ppo_epoch 15 --gain 0.01 --lr 1e-4 --use_wandb
    echo "training is done!"
done
