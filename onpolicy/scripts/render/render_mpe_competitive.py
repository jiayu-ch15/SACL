#!/usr/bin/env python
from pathlib import Path
import numpy as np
import os
import setproctitle
import sys
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.runner.competitive.mpe_runner import MPERunner as Runner


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                raise NotImplementedError(f"Unsupported env name: {all_args.env_name}, set env_name = MPE.")
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    # training args
    parser.add_argument("--competitive", action="store_true", default=False, help="by default False, use competitive runner.")
    parser.add_argument("--training_mode", type=str,default="self_play", choices=["self_play", "red_br", "blue_br"])
    parser.add_argument("--save_ckpt_interval", type=int, default=250, help="checkpoint save intervel")
    parser.add_argument("--red_model_dir", type=str, default=None, help="by default None, directory of fixed red model")
    parser.add_argument("--blue_model_dir", type=str, default=None, help="by default None, directory of fixed blue model")
    parser.add_argument("--red_valuenorm_dir", type=str, default=None, help="by default None, directory of red value norm model")
    parser.add_argument("--blue_valuenorm_dir", type=str, default=None, help="by default None, directory of blue value norm model")

    parser.add_argument("--fixed_valuenorm", action="store_true", default=False, help="by default False, use fixed value norm")

    # env args
    parser.add_argument("--scenario_name", type=str, default="simple_tag_corner", help="which scenario to run on")
    parser.add_argument("--horizon", type=int, default=200, help="environment horizon")
    parser.add_argument("--corner_min", type=float, default=2.0, help="minimum distance of corner")
    parser.add_argument("--corner_max", type=float, default=3.0, help="maximum distance of corner")
    parser.add_argument("--num_adv", type=int, default=3, help="number of adversarial agents")
    parser.add_argument("--num_good", type=int, default=1, help="number of good agents")
    parser.add_argument("--num_landmarks", type=int, default=2, help="number of landmarks")
    parser.add_argument("--hard_boundary", action="store_true", default=False, help="by default False, use hard boundary")
    parser.add_argument("--warm_up", type=int, default=0, help="timestep of warmup")

    parser.add_argument("--oppenent_name", type=str,
                        default='mappo', choices=["matrpo", "happo", "mat", "rmappo", "mappo", "rmappg", "mappg", "ft_rrt", "ft_nearest", "ft_apf", "ft_utility"])

    all_args = parser.parse_known_args(args)[0]
    all_args.num_agents = all_args.num_adv + all_args.num_good
    all_args.red_model_dir = all_args.red_model_dir.split(" ") if all_args.red_model_dir is not None else None
    all_args.blue_model_dir = all_args.blue_model_dir.split(" ") if all_args.blue_model_dir is not None else None
    all_args.red_valuenorm_dir = all_args.red_valuenorm_dir.split(" ") if all_args.red_valuenorm_dir is not None else None
    all_args.blue_valuenorm_dir = all_args.blue_valuenorm_dir.split(" ") if all_args.blue_valuenorm_dir is not None else None

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError(f"Unsupported algorithm name: {all_args.algorithm_name}.")

    assert (all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener") == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    assert all_args.use_render, ("use_render should be True")
    assert (all_args.red_model_dir is not None and all_args.blue_model_dir is not None), ("red_model_dir and blue_model_dir should be set")
    assert all_args.n_rollout_threads == 1, ("n_rollout_threads should be 1")
        
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

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [int(str(folder.name).split("run")[1]) for folder in run_dir.iterdir() if str(folder.name).startswith("run")]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    num_red = all_args.num_adv
    num_blue = all_args.num_good

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "num_red": num_red,
        "num_blue": num_blue,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    assert all_args.competitive == True, ("add --competitive to use competitive runner.")
    runner = Runner(config)
    runner.render()
    
    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
