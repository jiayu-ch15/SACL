#!/bin/sh
env="Agar"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=2 python eval_agar.py --env_name ${env} --seed ${seed} --num_agents 2 --episode_length 10 --model_dir "/home/yuchao/project/mappo-sc/results/Agar/agar_rollout16_batch16_length500_lr3e-4_attn1648_share_bad/" --eval_episodes 100 --save_gifs --eval
done
