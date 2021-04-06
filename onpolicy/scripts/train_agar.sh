#!/bin/sh
env="Agar"
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
     CUDA_VISIBLE_DEVICES=3 python train/train_agar.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --n_training_threads 2 --n_rollout_threads 128 --num_agents 2 --episode_length 128 --num_env_steps 60000000 --hidden_size 128 --attn_N 1 --attn_size 64 --attn_heads 4 --lr 2.5e-4 --user_name ethanyang --layer_N 1 --ppo_epoch 4 --clip_param 0.1 --num_mini_batch 1 --entropy_coef 0.01 --max_grad_norm 20 --gamma 0.995 --data_chunk_length 32 --eval_episodes 10 --eval_interval 20 --n_eval_rollout_threads 50 --use_cat_self False --use_single_network True --use_attn --use_eval True --share_policy False 
    echo "training is done!"
done
