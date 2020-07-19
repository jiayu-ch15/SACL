#!/bin/sh
env="StarCraft2"
map="3m"
algo="3m_lr5e-4_batch1_huber10_ppo4_entropy0.01_popart_norm"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 1 --ppo_epoch 4 --use_popart --episode_length 400 --lr 5e-4 --value_loss_coef 1 --num_env_steps 10000000 --data_chunk_length 10 --use_huber_loss --huber_delta 10 --entropy_coef 0.01 --attn_heads 4 --dropout 0.0 --use_feature_normlization
    echo "training is done!"
done
