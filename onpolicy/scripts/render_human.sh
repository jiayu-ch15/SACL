#!/bin/sh
env="Human"
scenario="simple_human"
num_landmarks=2
num_good_agents=1
num_adversaries=1
algo="rmappo"
exp="render"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_human.py \
    --use_human_command --use_fixed_prey --use_pos_four_direction --use_direction_reward --add_direction_encoder --use_all_reach --use_noisy_command \
    --direction_alpha 1 --view_threshold 0.5 --noisy_level 9 \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} \
    --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} --save_gifs \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --model_dir "/home/huangruixin/render/MPE/fixed_four_dirrew_encoder_1predator_1prey_obsmask_alpha1_view0.5_NoisyLevel9"
done
