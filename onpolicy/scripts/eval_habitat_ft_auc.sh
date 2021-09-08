#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=1
algo="ft_nearest" # ft_rrt ft_utility ft_apf  
exp="new_transform_global_apf_local_fmm_eval43"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_habitat_ft_auc.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --use_same_scene --scene_id 61 --eval_episodes 5 --use_eval --ifi 0.01 --seed 2 \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 45 \
    --num_local_steps 15 --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 \
    --critic_lr 2.5e-5 --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,1,1' \
    --hidden_size 256 --log_interval 1 --use_recurrent_policy --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --use_complete_reward \
    --use_centralized_V --use_delta_reward --wandb_name "mapping" --use_merge --user_name "yang-xy20" \
    --use_max --use_max_map --local_planner fmm --use_merge_local --use_different_start_pos --use_fixed_start_pos
    echo "evaluation is done!" 
done