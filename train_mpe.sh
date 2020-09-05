#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
#algo="simple_spread_parallel128_length50_nogru"
algo="test"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python train_mpe.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --share_policy --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 128 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000000 --layer_N 2 --ppo_epoch 15 --recurrent_policy
    echo "training is done!"
done
