#!/bin/sh
env="StarCraft2"
map="3s_vs_4z"
algo="rmappo"
exp="clean_state_xy_hidden512_N2"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 50000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --user_name "zoeyuchao" --use_state_agent --hidden_size 512 --layer_N 2
    echo "training is done!"
done
