#!/bin/sh
env="Human"
scenario="simple_human"
num_landmarks=2
num_good_agents=1
num_adversaries=1
algo="rmappo"
exp="fixed_four_dirrew_encoder_1predator_1prey_obsmask_alpha1_view0.5_NoisyLevel9"
seed_max=1
ulimit -n 234567

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_human.py --direction_alpha 1.0 --view_threshold 0.5 --noisy_level 9 \
    --use_noisy_command --use_fixed_prey --use_pos_four_direction --add_direction_encoder --use_direction_reward \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} \
    --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} \
    --seed 1 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 \
    --num_env_steps 18000000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --wandb_name "zoeyuchao" --user_name "zoeyuchao" --use_eval 
done