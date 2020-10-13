#!/bin/sh
env="StarCraft2"
map="3m"
algo="check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=6 python train_sc_state.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed_max} --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 --gain 1 --use_value_high_masks --eval --attn
echo "training is done!"
