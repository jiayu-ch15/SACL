#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="new_10scenes_same_loc_rand_rot_middle_add_agents_orientation_full_map_ppo4_eval16"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,2 python eval/eval_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train" --use_same_scene --scene_id 16 --eval_episodes 2 --use_render --save_gifs --ifi 0.01 --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 300 --num_local_steps 15 --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' --hidden_size 256 --log_interval 1 --use_recurrent_policy  --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --model_dir "./results/Habitat/mappo/new_10scenes_same_loc_rand_rot_middle_add_agents_orientation_full_map_ppo4/wandb/run-20210513_201848-a41ezq4m/files/global_actor_best.pt" --use_intrinsic_reward --use_complete_reward --use_centralized_V --use_delta_reward --wandb_name "mapping" --use_wandb --use_random_rotation
    echo "training is done!" 
done
