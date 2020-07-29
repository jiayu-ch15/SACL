#!/bin/sh
env="Agar"
algo="test"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train_agar.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 2 --num_mini_batch 1 --ppo_epoch 4 --use_popart --num_agents 2 --episode_length 10 --lr 3e-4 --value_loss_coef 1 --num_env_steps 40000000 --data_chunk_length 10 --use_huber_loss --huber_delta 10 --entropy_coef 0.01 --use-max-grad-norm --max-grad-norm 20.0 --clip_param 0.2 --share_reward --attn
    echo "training is done!"
done
