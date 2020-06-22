#!/bin/sh
env="single_navigation"
seed_max=5
length=10

echo "env is ${env}"

for seed in `seq ${seed_max}`
do 
    let "step_unknown=seed*10"
    algo="lstm"+${step_unknown}
    echo "step_unknown is ${step_unknown},algo is ${algo}:"
    CUDA_VISIBLE_DEVICES=2 python train_MPE.py --env_name ${env} --algorithm_name ${algo} --seed 1 --n_rollout_threads 512 --num_mini_batch 1 --num_agents 1 --num_landmarks 3 --ppo_epoch 10 --episode_length 60 --step_unknown ${step_unknown} --lr 2e-3 --value_loss_coef 0.5 --entropy_coef 0.01 --num_env_steps 100000000 --recurrent_policy --data_chunk_length ${length}
done
