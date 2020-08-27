#!/bin/sh
env="StarCraft2"
map="3m"
algo="ablation_3m_mini16_ppo5"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_sc.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 16 --episode_length 400 --num_env_steps 5000000 --ppo_epoch 5
    echo "training is done!"
done
