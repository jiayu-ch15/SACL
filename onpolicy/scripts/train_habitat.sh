#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="no_penatlty_reward_10scenes_same_loc_rand_rot_middle_add_agents_abs_orientation_version3_full_map_ppo5"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6,7,0 python train/train_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train" --seed 3 --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 5 --num_local_steps 15 --max_episode_length 300 --num_env_steps 3000000 --ppo_epoch 5 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10 --use_complete_reward --eval_episodes 1 --use_centralized_V --wandb_name "mapping"  --use_name "yang-xy20" --use_intrinsic_reward --use_selected_middle_scenes --use_delta_reward --use_random_rotation --use_abs_orientation
    echo "training is done!"
done