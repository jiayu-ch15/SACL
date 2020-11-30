#!/bin/sh
env="MPE"
scenario_name="simple_spread"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python eval_mpe.py --env_name ${env} --seed ${seed} --scenario_name ${scenario_name} --episode_length 70 --eval_episodes 2 --model_dir "/home/yuchao/project/mappo-sc/results/MPE/simple_spread/simple_spread_parallel128_length70_nolstm/"
done
