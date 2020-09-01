#!/bin/sh
env="StarCraft2"
map="bane_vs_bane"
algo="fix_bane_vs_bane_parallel2"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train_sc.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 2 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 --eval
    echo "training is done!"
done
