#!/bin/sh
env="box_blocking"
scenario_name="quadrant"
num_agents=1
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python eval_hns.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --num_agents ${num_agents} --model_dir "/home/yuchao/project/mappo-sc/results/BoxLocking/quadrant/quadrant_parallel1000_length160_attn/"
done
