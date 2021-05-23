#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmappo"
exp="renderßß"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python render/render_smac.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed 1 \
    --n_training_threads 1 --render_episodes 1 --n_rollout_threads 1 --num_mini_batch 1 \
    --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 \
    --use_value_active_masks --use_render --use_wandb --user_name "zoeyuchao" --model_dir "/home/tsing69/project/onpolicy/onpolicy/scripts/results/StarCraft2/3m/rmappo/test/run2/models/"
done
