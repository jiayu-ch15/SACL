#!/bin/sh
env="Agar"
algo="agar_rollout16_batch16_length500_kill1_coop1_action2"
seed_max=1

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train_agar.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 1 --num_mini_batch 1 --ppo_epoch 4 --use_popart --num_agents 2 --episode_length 10 --lr 5e-4 --value_loss_coef 1 --num_env_steps 40000000 --data_chunk_length 10 --use_huber_loss --huber_delta 10 --entropy_coef 0.005 --use-max-grad-norm --max-grad-norm 20.0 --clip_param 0.1
    echo "training is done!"
done
