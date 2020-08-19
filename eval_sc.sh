#!/bin/sh
env="StarCraft2"
map="bane_vs_bane"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python eval_sc.py --env_name ${env} --seed ${seed} --map_name ${map} --model_dir "/home/yuchao/project/mappo-sc/results/StarCraft2/bane_vs_bane/ok_bane_vs_bane_parallel2/" --cuda
done
