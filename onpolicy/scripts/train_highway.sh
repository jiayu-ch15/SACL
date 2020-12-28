#!/bin/sh
env="Highway"
scenario="highway-v0"
num_agents=3
n_attacker=1
n_npc=0
algo="rmappo"
exp="render"
seed_max=1
ulimit -n 22222

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --n_attacker ${n_attacker} --n_npc ${n_npc} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --episode_length 40 --log_interval 1 --use_wandb
    echo "training is done!"
done
