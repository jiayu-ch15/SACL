#!/bin/sh
env="Habitat"
num_agents=1
algo="mappo"
exp="mlp"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_habitat.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --seed 3 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_recurrent_policy --use_maxpool2d --cnn_layers_params [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1), (64, 3, 1, 1), (32, 3, 1, 1)]
    echo "training is done!"
done
