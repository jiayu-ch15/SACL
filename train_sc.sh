#!/bin/sh
env="StarCraft2"
map="3m"
algo="3m_batch1_length800_noclipgrad_lr1e-4_nolrdecay_weightdecay1e-8"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 1 --ppo_epoch 4 --episode_length 800 --lr 1e-4 --value_loss_coef 1 --num_env_steps 10000000 --data_chunk_length 10 --use_clipped_value_loss
done
