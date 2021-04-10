#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2,3,6 python train/train_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train" --seed 3 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 10 --num_local_steps 15 --max_episode_length 300 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 1 --use_complete_reward --use_restrict_map --use_same_scene --scene_id 7 --eval_episodes 1 --use_intrinsic_reward --wandb_name "mapping" --use_wandb --use_different_start_pos --use_same_rotation
    echo "training is done!"
done
