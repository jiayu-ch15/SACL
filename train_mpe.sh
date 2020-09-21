#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
algo="gru/simple_speaker_listener_parallel128_length25_lr5e-4"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_mpe.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 10000000 --ppo_epoch 15 --lr 5e-4 --gain 0.01 --share_policy
    echo "training is done!"
done
