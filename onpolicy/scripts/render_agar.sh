#!/bin/sh
env="Agar"
scenario="simple_spread"
#share_reward=True
use_curriculum_learning=False
action_repeat=5
num_agents=2
algo="rmappo"
exp="render"
seed_max=1
model_dir=

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_agar.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --model_dir ${model_dir} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 128 --render_episodes 1 --use_wandb  --save_gifs --hidden_size 128 --attn_N 1 --attn_size 64 --attn_heads 4 --layer_N 1 --use_cat_self False --use_attn #--share_policy --use_single_network
done