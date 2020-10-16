#!/bin/sh
env="Agar"
algo="new_agar_rollout24_batch16_length600_kill0_coop1_hidden128_attn11284_15e6_lr25e-4_curriculum"
#algo="test"
seed_max=1

echo "env is ${env}, algo is ${algo}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=4 python train/train_agar.py --env_name ${env} --algorithm_name ${algo} --seed ${seed_max} --n_training_threads 1 --n_rollout_threads 24 --num_mini_batch 16 --num_agents 2 --episode_length 600 --num_env_steps 40000000 --hidden_size 128 --attn --attn_N 1 --attn_size 128 --attn_heads 4 --lr 2.5e-4 --cuda
echo "training is done!"
