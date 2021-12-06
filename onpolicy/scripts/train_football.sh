#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="test_shared_epoch_5"

# football param
num_agents=3
representation="simple115v2"
# write_goal_dumps
# dump_frequency=10
# log_dir="football_log"

# train param
num_env_steps=100000000
episode_length=200
n_rollout_threads=1000 # 1000
ppo_epoch=5 # 5, 10, 15
num_mini_batch=2 # 2, 4
data_chunk_length=10 # 10, 20

# eval param
eval_interval=10 # ?
eval_episodes=100
n_eval_rollout_threads=100 # 100

seed_max=1

echo "env: ${env} \t scenario: ${scenario} \t algo: ${algo} \t exp: ${exp} \t max seed: ${seed_max}"
for seed in `seq ${seed_max}`
do
    echo "seed ${seed}"
    CUDA_VISIBLE_DEVICES=5 python train/train_football.py \
    --env_name ${env} --scenario_name ${scenario} \
    --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
    --num_agents ${num_agents} --representation ${representation} \
    --num_env_steps ${num_env_steps} \
    --n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} \
    --episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
    --data_chunk_length ${data_chunk_length} \
    --use_eval \
    --eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
    --n_eval_rollout_threads ${n_eval_rollout_threads} \
    --user_name "zelaix" --wandb_name "zelaix"
done
