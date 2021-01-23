#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
exp="mlp_critic1e-3_entropy0.015"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1000 --n_eval_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --user_name zoeyuchao --use_recurrent_policy --model_dir "/home/yuchao/project/onpolicy/onpolicy/scripts/results/Hanabi/Hanabi-Full/mappo/mlp_critic1e-3_entropy0.015/wandb/run-20210119_220854-1tkmu1s7/files/" --entropy_coef 0.015
    echo "training is done!"
done
