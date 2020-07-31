#!/bin/sh
env="Agar"
algo="agar_rollout24_batch16_length500_kill0_coop1_hidden128_right"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_agar.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 24 --num_mini_batch 16 --num_agents 2 --episode_length 500 --num_env_steps 40000000 --hidden_size 128 
    echo "training is done!"
done
