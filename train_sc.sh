#!/bin/sh
env="StarCraft2"
map="2c_vs_64zg"
algo="2c_vs_64zg_episode800_lr7e-4_batch16_attn2648_huber10_entropy0.007"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 16 --ppo_epoch 7 --episode_length 800 --lr 7e-4 --value_loss_coef 1 --num_env_steps 10000000 --data_chunk_length 10 --attn --attn_N 2 --use_huber_loss --huber_delta 10 --entropy_coef 0.007
    echo "training is done!"
done
