#!/bin/sh
env="StarCraft2"
map="3m"
algo="check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 5000000 --ppo_epoch 15 --gain 1 --use_value_active_masks --eval
    echo "training is done!"
done
