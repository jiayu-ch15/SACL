#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=1
algo="mappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4,5 python eval/eval_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --eval_episodes 1 --use_render --ifi 0.01 --save_gifs --split "train" --use_wandb --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 10 --episode_length 40 --max_episode_length 50 --num_env_steps 20000000 --ppo_epoch 10 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --model_dir "./results/Habitat/mappo/debug/run89/models/"
    echo "training is done!"
done
