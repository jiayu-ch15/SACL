#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="ft_rrt" 
exp="ft_rrt_grid19_rand_loc "
seed_max=1
#ft_utility,ft_nearest,ft_apf
echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}"
    CUDA_VISIBLE_DEVICES=3 python render/render_gridworld_ft.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_eval --render_episodes 50 \
      --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' \
      --ifi 0.3 --max_steps 100 --grid_size 19 --use_local --use_random_pos --agent_view_size 7 --use_single --use_merge --wandb_name "mapping" --user_name "yang-xy20" 
done 