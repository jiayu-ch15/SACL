#!/bin/sh
env="HideAndSeek"
scenario_name="quadrant"
num_seekers=2
num_hiders=2
num_boxes=2
num_ramps=1
num_food=0
algo="default_quadrant_seeker2_hider2_box2_ramp1"
seed_max=1

echo "env is ${env}, scenario is ${scenario_name}, num_seekers is ${num_seekers}, num_hiders is ${num_hiders}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train_hns.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario_name} --num_seekers ${num_seekers} --num_hiders ${num_hiders} --num_boxes ${num_boxes} --num_ramps ${num_ramps} --num_food ${num_food} --seed ${seed} --n_rollout_threads 600 --num_mini_batch 2 --episode_length 800 --num_env_steps 500000000 --ppo_epoch 15 --attn --hidden_size 256 --attn_size 128
    echo "training is done!"
done
