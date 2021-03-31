#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="baseline"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1,2 python train/train_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --use_wandb --seed 1 --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 10 --num_local_steps 20 --max_episode_length 400 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 10 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10
    echo "training is done!"
done
