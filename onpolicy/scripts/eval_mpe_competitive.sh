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
model_name=("sp" "sacl-unif" "sacl-var" "sacl-var**5" "sacl-rb_var" "sacl-rb_var**5")
model_dir=(
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_uniform-sp/wandb/run-20230405_072147-2w2sddz3/files"
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-uniform-wo_time/wandb/run-20230405_143444-2hngjx8z/files"
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-variance-wo_time/wandb/run-20230405_143758-4yi5jvmo/files"
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-variance**5-wo_time/wandb/run-20230405_144611-lji84yjt/files"
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance-wo_time/wandb/run-20230405_144206-32c37840/files"
    "/home/zelaix/projects/onpolicy/onpolicy/scripts/results/MPE/simple_tag_corner/mappo/2_corner-sacl-rb_variance**5-wo_time/wandb/run-20230405_144713-28h7cvs6/files"
)

# user name
user_name="zelaix"
wandb_name="sacl"


for i in {0..5}; do
    for j in {0..5}; do
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
