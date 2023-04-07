#!/bin/bash
# exp config
exp="eval"
algo="mappo"
seed=0
# env config
env="MPE"
scenario="simple_tag_corner"
horizon=200
corner_min=1.0
corner_max=2.0
num_adv=3
num_good=1
num_landmarks=2
# policy config
attn_size=32
# train config
n_rollout_threads=1
# eval config
n_eval_rollout_threads=100
eval_episodes=100
model_name=("sacl-unif" "sacl-bias" "sacl-var")
model_dir=(
    "/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-random/wandb/run-20230406_131114-3hqxl9jp/files"
    "/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-variance_add_bias_beta0_alpha1/wandb/run-20230406_082127-fq6vsove/files"
    "/home/jiayu-ch15/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-variance_add_bias_beta1_alpha0/wandb/run-20230406_081708-1fqtxi08/files"
)

# user name
user_name="chenjy"
wandb_name="sacl"


for i in {0..2}; do
    for j in {0..2}; do
        echo "======= ${model_name[i]} (adv) vs ${model_name[j]} (good) ======="

        CUDA_VISIBLE_DEVICES=5 python eval/eval_mpe_competitive.py \
        --experiment_name ${exp} --algorithm_name ${algo} --seed ${seed} --competitive \
        --env_name ${env} --scenario_name ${scenario} --horizon ${horizon} \
        --corner_min ${corner_min} --corner_max ${corner_max} \
        --num_adv ${num_adv} --num_good ${num_good} --num_landmarks ${num_landmarks} \
        --use_attn --attn_size ${attn_size} --use_recurrent_policy \
        --n_rollout_threads ${n_rollout_threads} \
        --use_eval --eval_episodes ${eval_episodes} \
        --n_eval_rollout_threads ${n_eval_rollout_threads} \
        --red_model_dir ${model_dir[i]} --blue_model_dir ${model_dir[j]} \
        --use_wandb
    done
done
