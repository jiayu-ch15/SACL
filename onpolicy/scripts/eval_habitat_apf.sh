#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="ft_apf"
exp="render_habitat_apf"
seed_max=1
my_seed=10
envs_max=60

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1 python eval/eval_habitat_apf.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train"  --eval_episodes 50 --use_eval --ifi 0.01 --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 300 --num_local_steps 15 --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "/home/gaojiaxuan/onpolicy/onpolicy/scripts/models/slam_best.pt" --load_local "/home/gaojiaxuan/onpolicy/onpolicy/scripts/models/local_best.pt" --use_complete_reward --use_centralized_V --use_delta_reward --wandb_name "mapping" --use_different_start_pos \
    --user_name "yang-xy20" \
    --use_ft_global --ft_global_mode apf \
    --use_same_scene --scene_id 43  --use_wandb \
    --use_render --save_gifs
    echo "training is done!" 
done
