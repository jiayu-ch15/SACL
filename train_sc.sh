#!/bin/sh
env="StarCraft2"
map="3m"
algo="3m_minibatchsize640"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 32 --num_mini_batch 1 --ppo_epoch 4 --episode_length 200 --lr 1e-3 --value_loss_coef 1 --num_env_steps 1000000
done