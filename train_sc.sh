#!/bin/sh
env="StarCraft2"
map="2s_vs_1sc"
#algo="win_2s3z_xavier_uniform_attn"
algo="test"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_sc.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 1 --num_mini_batch 1 --episode_length 20 --num_env_steps 10000000 --ppo_epoch 15 --use_clipped_value_loss --cuda 
    echo "training is done!"
done
