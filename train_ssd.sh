#!/bin/sh
env="Cleanup"
algo="Cleanup_rollout128_batch16_ppo4_epilength500_popart_norm"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_ssd.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 64 --num_mini_batch 16 --ppo_epoch 4 --use_popart --episode_length 500 --lr 5e-4 --value_loss_coef 1 --num_env_steps 10000000 --data_chunk_length 10 --use_huber_loss --huber_delta 10 --entropy_coef 0.01 --use_feature_normliazation
    echo "training is done!"
done
