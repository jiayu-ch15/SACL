#!/bin/sh
env="Agar"
num_agents=2
algo="rmappo"
exp="render"
seed_max=1
model_dir="/home/tsing88/onpolicy/onpolicy/scripts/results/Agar/rmappo/debug/wandb/run-20210429_193003-3lmtyj20/files"

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_agar.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --model_dir ${model_dir} --n_training_threads 1 --n_rollout_threads 1 --use_render --render_episodes 10 --use_wandb  --hidden_size 128 --attn_N 1 --attn_size 64 --attn_heads 4 --layer_N 1 --use_cat_self --use_attn --use_centralized_V --share_policy #--use_single_network --save_gifs
done