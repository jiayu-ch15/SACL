#!/bin/sh
env="StarCraft2"
map="3m"
algo="3m_minibatchsize32_attn"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 32 --num_mini_batch 32 --ppo_epoch 4 --episode_length 200 --lr 7e-4 --value_loss_coef 1 --num_env_steps 100000000 --data_chunk_length 10 --attn
done
