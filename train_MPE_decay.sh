#!/bin/sh
 CUDA_VISIBLE_DEVICES=1 python train_MPE.py --env_name "single_navigation" --algorithm_name "test_decay" --seed 1 --n_rollout_threads 512 --num_mini_batch 1 --num_agents 1 --num_landmarks 3 --ppo_epoch 10 --episode_length 60 --step_unknown 20 --lr 2e-3 --value_loss_coef 0.5 --entropy_coef 0.01 --num_env_steps 100000000 --recurrent_policy --data_chunk_length 10 --unknown_decay --decay_episode 100

