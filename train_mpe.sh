#!/bin/sh
env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="mlp/simple_reference_parallel128_length25"
seed_max=10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train_mpe.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 10000000 --ppo_epoch 15  --gain 0.01 --recurrent_policy
    echo "training is done!"
done
