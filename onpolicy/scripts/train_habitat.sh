#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
#exp="new_time_penalty_centralized_V_merge_partial_reward_all_rand_rot_pos_center_single_local_ppo4"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_habitat.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --seed 2 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 \
    --num_local_steps 15 --max_episode_length 300 --num_env_steps 3000000 --ppo_epoch 4 --gain 0.01 \
    --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d \
    --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 \
    --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10 \
    --eval_episodes 1 --use_selected_middle_scenes \
    --use_delta_reward --use_different_start_pos --use_complete_reward --use_merge_partial_reward \
    --use_center --use_merge --use_time_penalty --use_same_scene --scene_id 16 \
    --wandb_name "mapping" --user_name "yuchao"
    echo "training is done!"
done