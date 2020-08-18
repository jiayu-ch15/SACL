#!/bin/sh
env="StarCraft2"
map="2c_vs_64zg"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=4 python eval_sc.py --env_name ${env} --seed ${seed} --map_name ${map} --model_dir "/home/yuchao/project/mappo-sc/results/StarCraft2/2c_vs_64zg/ok_2c_vs_64zg/" --cuda
done
