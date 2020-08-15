#!/bin/sh
env="StarCraft2"
map="2s3z"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=7 python eval_sc.py --env_name ${env} --seed ${seed} --map_name ${map} --episode_length 400 --model_dir "/home/yuchao/project/mappo-sc/results/StarCraft2/2s3z/win_2s3z_orth/" --eval_episodes 100 --cuda
done
