#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="new_no_orientation_nofc_time_penalty_merge_partial_reward_rand55_rot_pos_center_merge_local_ppo4"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1,2,0 python train/train_habitat.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --seed 1 --n_training_threads 1 --n_rollout_threads 9 --num_mini_batch 5 \
    --num_local_steps 15 --max_episode_length 300 --num_env_steps 3000000 --ppo_epoch 4 --gain 0.01 \
    --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d \
    --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 \
    --use_recurrent_policy   \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10 \
    --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" \
    --eval_episodes 1 --use_selected_middle_scenes\
    --use_delta_reward --use_different_start_pos --use_complete_reward --use_merge_partial_reward \
    --use_center --use_merge --use_time_penalty --wandb_name "mapping" --user_name "yang-xy20"\
    --slam_keys rgb --use_max --use_max_map --use_merge_local
    echo "training is done!"
done
