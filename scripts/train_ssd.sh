#!/bin/sh
env="Harvest"
algo="Harvest_rollout8_batch16_popart_individual"
#algo="test"
seed_max=1

echo "env is ${env}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_ssd.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 16 --ppo_epoch 4 --use_popart --episode_length 1000 --lr 5e-4 --value_loss_coef 1 --num_env_steps 1000000 --data_chunk_length 10 --use_huber_loss --huber_delta 10 --entropy_coef 0.01 
    echo "training is done!"
done
