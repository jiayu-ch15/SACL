#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.football.Football_Env import FootballEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = FootballEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='academy_3_vs_1_with_keeper', 
                        help="which scenario to run on.")
    parser.add_argument("--stacked", action='store_true', default=False, 
                        help="by default False. If True, stack 4 observations.")
    parser.add_argument("--representation", type=str, default="simple115v2", 
                        choices=["simple115v2", "extracted", "pixels_gray", 
                                 "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring", 
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--write_goal_dumps", action='store_true', 
                        default=False, 
                        help="by default False. If True, dump traces up to 200 "
                             "frames before goals.")
    parser.add_argument("--write_full_episode_dumps", action='store_true', 
                        default=False, 
                        help="by default False. If True, dump traces for every "
                             "episode.")
    parser.add_argument("--render", action='store_true', 
                        default=False, 
                        help="by default False. If True, enable render ")
    parser.add_argument("--dump_frequency", type=int, default=1,
                        help="how often to write dumps/videos in terms of "
                             "episodes (default: 1)")
    parser.add_argument("--log_dir", type=str, default="", 
                        help="log directory.")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="number of controlled players.")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.football_runner import FootballRunner as Runner
    else:
        from onpolicy.runner.separated.football_runner import FootballRunner as Runner

    runner = Runner(config)
    runner.render()
    
    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
