#!/bin/sh
env="StarCraft2"
map="3m"
algo="attn/3m_layernorm"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_sc_state.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 --gain 1 --use_value_high_masks --eval --attn --attn_size 128
    echo "training is done!"
done
