#!/bin/sh
env="StarCraft2"
map="2s3z"
algo="2s3z_minibatchsize300_epilength500"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 6 --num_mini_batch 1 --ppo_epoch 4 --episode_length 500 --lr 1e-3 --value_loss_coef 1 --num_env_steps 10000000 --data_chunk_length 10
done