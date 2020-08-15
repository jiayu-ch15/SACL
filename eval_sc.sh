#!/bin/sh
env="StarCraft2"
map="3m"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=7 python eval_sc.py --env_name ${env} --seed ${seed} --map_name ${map} --episode_length 400 --model_dir "/home/yuchao/project/mappo-sc/results/StarCraft2/3m/new_3m/" --eval_episodes 32 --cuda
done
