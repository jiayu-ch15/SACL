#!/usr/bin/env python
import sys
import os
import time
import copy
import glob
import shutil
import numpy as np
import itertools
import wandb
from tensorboardX import SummaryWriter
import socket
import setproctitle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config

from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer
from envs.decentralized_bottlenecks.flow.multiagent_envs.multi_bottleneck_env import MultiBottleneckEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController


def make_parallel_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MultiBottleneckEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MultiBottleneckEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def setup_flow_params(args):
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    av_frac = args.av_frac
    if args.lc_on:
        lc_mode = 1621
    else:
        lc_mode = 0

    vehicles = VehicleParams()
    if not np.isclose(av_frac, 1):
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=lc_mode,
            ),
            num_vehicles=1)
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)
    else:
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(            
                lane_change_mode=0,
            ),
            num_vehicles=1)

    # flow rate
    flow_rate = 1900 * args.scaling

    controlled_segments = [('1', 1, False), ('2', 2, True), ('3', 2, True),
                           ('4', 2, True), ('5', 1, False)]
    num_observed_segments = [('1', 1), ('2', 3), ('3', 3), ('4', 3), ('5', 1)]
    if np.isclose(args.av_frac, 0.4):
        q_init = 1000
    else:
        q_init = 600
    additional_env_params = {
        'target_velocity': 40,
        'disable_tb': True,
        'disable_ramp_metering': True,
        'controlled_segments': controlled_segments,
        'symmetric': False,
        'observed_segments': num_observed_segments,
        'reset_inflow': True,
        'lane_change_duration': 5,
        'max_accel': 3,
        'max_decel': 3,
        'inflow_range': [args.low_inflow, args.high_inflow],
        'start_inflow': flow_rate,
        'congest_penalty': args.congest_penalty,
        'communicate': args.communicate,
        "centralized_obs": args.central_obs, # ! attn
        "aggregate_info": args.aggregate_info, # !
        "av_frac": args.av_frac,
        "congest_penalty_start": args.congest_penalty_start,
        "lc_mode": lc_mode,
        "life_penalty": args.life_penalty,
        'keep_past_actions': args.keep_past_actions,
        "num_sample_seconds": args.num_sample_seconds,
        "speed_reward": args.speed_reward,
        'fair_reward': False,  # This doesn't do anything, remove
        'exit_history_seconds': 0,  # This doesn't do anything, remove
        'reroute_on_exit': args.reroute_on_exit,

        # parameters for the staggering controller that we imitate
        "n_crit": 8,
        "q_max": 15000,
        "q_min": 200,
        "q_init": q_init, #
        "feedback_coeff": 1, #
        'num_imitation_iters': args.num_imitation_iters,

        # parameters from imitation
        "simple_env": args.simple_env,
        "super_simple_env": args.super_simple_env,

        # curriculum stuff
        "curriculum": args.curriculum,
        "num_curr_iters": args.num_curr_iters,
        "min_horizon": args.min_horizon,
        "horizon": args.horizon,
        "rew_n_crit": args.rew_n_crit,

        "no_congest_number": args.no_congest_number,
    }

    if args.dqfd:
        additional_env_params.update({
            "num_expert_steps": args.num_expert_steps,
            "action_discretization": 5,
            "fingerprinting": args.fingerprinting
        })

    # percentage of flow coming out of each lane
    inflow = InFlows()
    if not np.isclose(args.av_frac, 1.0):
        inflow.add(
            veh_type='human',
            edge='1',
            vehs_per_hour=flow_rate * (1 - args.av_frac),
            departLane='random',
            departSpeed=23.0)
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate * args.av_frac,
            departLane='random',
            departSpeed=23.0)
    else:
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate,
            departLane='random',
            departSpeed=23.0)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id='2')
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id='3')

    additional_net_params = {'scaling': args.scaling, "speed_limit": 23.0}

    if args.imitate:
        env_name = 'MultiBottleneckImitationEnv'
    elif args.dqfd:
        env_name = 'MultiBottleneckDFQDEnv'
    else:
        env_name = 'MultiBottleneckEnv'

    # if args.super_simple_env:
    #     scenario = 'SimpleBottleneckScenario'
    # else:
    scenario='BottleneckScenario'
    warmup_steps = 0
    if args.reroute_on_exit:
        warmup_steps = int(300 / args.sims_per_step)

    flow_params = dict(
        # name of the experiment
        exp_tag=args.exp_title,

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the scenario class the experiment is running on
        scenario=scenario,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=args.sim_step,
            render=args.render,
            print_warnings=False,
            restart_instance=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=int(warmup_steps / args.sim_step),
            sims_per_step=args.sims_per_step,
            horizon=args.horizon,
            clip_actions=False,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # scenario's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.vehicles.Vehicles)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='uniform',
            min_gap=5,
            lanes_distribution=float('inf'),
            edges_distribution=['2', '3', '4', '5'],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.traffic_lights.TrafficLights)
        tls=traffic_lights,
    )
    return flow_params

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    # arguments for flow
    parser.add_argument('--sims_per_step', type=int, default=1, help='How many steps to take per action')
    parser.add_argument('--render', action='store_true', help='Show sumo-gui of results')
    parser.add_argument('--horizon', type=int, default=1000, help='Horizon of the environment')
    parser.add_argument('--sim_step', type=float, default=0.5, help='dt of a timestep')
    parser.add_argument('--low_inflow', type=int, default=800, help='the lowest inflow to sample from')
    parser.add_argument('--high_inflow', type=int, default=2200, help='the highest inflow to sample from')
    parser.add_argument('--av_frac', type=float, default=0.1, help='What fraction of the vehicles should be autonomous.'
        'If < 0, will be trained on random penetrations between 5% and 40%')
    parser.add_argument('--scaling', type=int, default=1, help='How many lane should we start with. Value of 1 -> 4, '
                                                               '2 -> 8, etc.')
    parser.add_argument('--lc_on', action='store_true', help='If true, lane changing is enabled.')
    parser.add_argument('--congest_penalty', action='store_true', help='If true, an additional penalty is added '
                                                                       'for vehicles queueing in the bottleneck')
    parser.add_argument('--communicate', action='store_true', help='If true, the agents have an additional action '
                                                                   'which consists of sending a discrete signal '
                                                                   'to all nearby vehicles')
    parser.add_argument('--central_obs', action='store_true', help='If true, all agents receive the same '
                                                                   'aggregate statistics')
    parser.add_argument('--aggregate_info', action='store_true', help='If true, agents receive some '
                                                                      'centralized info')
    parser.add_argument('--congest_penalty_start', type=int, default=30, help='If congest_penalty is true, this '
                                                                              'sets the number of vehicles in edge 4'
                                                                              'at which the penalty sets in')
    parser.add_argument('--life_penalty', type=float, default=0, help='How much to subtract in the reward at each '
                                                                     'time-step for remaining in the system.')
    parser.add_argument('--keep_past_actions', action='store_true', help='If true we append the agents past actions '
                                                                         'to its observations')
    parser.add_argument('--num_sample_seconds', type=float, default=0.5,
                        help='How many seconds back in time the outflow reward should sample over. It defaults to '
                             'only looking at the current step')
    parser.add_argument('--no_congest_number', action='store_true', default=False,
                        help='Remove congestion number from the state.')
    parser.add_argument('--speed_reward', action='store_true', default=False,
                        help='If true the reward is the mean AV speed. If not set the reward is outflow')
    parser.add_argument('--imitate', action='store_true', default=False,
                        help='If true, the first 30 iterations are supervised learning on imitation of an IDM vehicle')
    parser.add_argument('--final_imitation_weight', type=float, default=0.0,
                        help='This is the value we decrease the imitation weight to after imitation is done.'
                             'Make it non-zero to prevent the policy from totally straying from the imitation weight.')
    parser.add_argument('--num_imitation_iters', type=int, default=50,
                        help='Number of iterations to imitate controller. must enable --imitate')
    parser.add_argument('--hard_negative_mining', action='store_true', default=False,
                        help='Use only the top 10 percent actions to imitate')
    parser.add_argument('--mining_frac', type=float, default=0.1,
                        help='The percentage of top scores to imitate on. .10 will imitate on the top 10%')
    parser.add_argument('--simple_env', action='store_true', default=False,
                        help='If true, the imitation env mimics the observations available to the imitated controllers')
    parser.add_argument('--super_simple_env', action='store_true', default=False,
                        help='If true, the imitation env mimics the observations available to the imitated controllers')
    parser.add_argument('--centralized_vf', action='store_true', default=False,
                        help='If true, use a centralized value function')
    parser.add_argument('--central_vf_size', type=int, default=64, help='The number of hidden units in '
                                                                        'the value function')
    parser.add_argument('--max_num_agents', type=int, default=120, help='The maximum number of agents we could ever have')
    parser.add_argument('--terminal_reward', action='store_true', default=False)
    parser.add_argument('--post_exit_rew_len', type=int, default=200, help='This is how much of the future outflow '
                                                                           'is given to the agent as a terminal reward')
    parser.add_argument('--rew_n_crit', type=int, default=0, help='If set to a value above zero, we get rewarded if fewer than '
                                                              'n_crit AVs in the bottleneck, and penalized if above')
    parser.add_argument('--reroute_on_exit', action='store_true', default=False,
                        help='Put back RL vehicles that have left')

    # dqfd arguments
    parser.add_argument('--dqfd', action='store_true', default=False,
                        help='Whether to use dqfd')
    parser.add_argument('--num_expert_steps', type=int, default=5e4, help='How many steps to let the expert take'
                                                                          'before switching back to the actor')
    parser.add_argument('--fingerprinting', action='store_true', default=False,
                        help='Whether to add the iteration number to the inputs')


    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    flow_params = setup_flow_params(all_args)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (
            all_args.recurrent_policy or all_args.naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.recurrent_policy and all_args.naive_recurrent_policy) == False, (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if "mappo" in all_args.algorithm_name:
        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    elif "mappg" in all_args.algorithm_name:
        if all_args.use_single_network:
            from algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
            from algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            from algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
            from algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

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

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        log_dir = str(run_dir / 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writter = SummaryWriter(log_dir)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_parallel_env(all_args)
    if all_args.eval:
        eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    # Policy network
    if all_args.use_obs_instead_of_state:
        cat_self = False
    else:
        cat_self = True
    if all_args.share_policy:
        if all_args.use_centralized_V:
            share_observation_space = envs.share_observation_space[0]
        else:
            share_observation_space = envs.observation_space[0]

        if all_args.model_dir == None or all_args.model_dir == "":
            policy = Policy(all_args,
                            envs.observation_space[0],
                            share_observation_space,
                            envs.action_space[0],
                            device=device,
                            cat_self=cat_self)
        else:
            policy = torch.load(
                str(all_args.model_dir) + "/agent_model.pt")['model']

        # algorithm
        trainer = TrainAlgo(all_args, policy, device=device)

        # replay buffer
        buffer = SharedReplayBuffer(all_args,
                                    num_agents,
                                    envs.observation_space[0],
                                    share_observation_space,
                                    envs.action_space[0])
    else:
        trainer = []
        buffer = []
        for agent_id in range(num_agents):
            if all_args.use_centralized_V:
                share_observation_space = envs.share_observation_space[agent_id]
            else:
                share_observation_space = envs.observation_space[agent_id]

            if all_args.model_dir == None or all_args.model_dir == "":
                po = Policy(all_args,
                            envs.observation_space[agent_id],
                            share_observation_space,
                            envs.action_space[agent_id],
                            device=device,
                            cat_self=cat_self)
            else:
                po = torch.load(str(all_args.model_dir) +
                                "/agent" + str(agent_id) + "_model.pt")['model']

            tr = TrainAlgo(all_args, po, device=device)
            bu = SeparatedReplayBuffer(all_args,
                                       envs.observation_space[agent_id],
                                       share_observation_space,
                                       envs.action_space[agent_id])
            trainer.append(tr)
            buffer.append(bu)

    # reset env
    obs = envs.reset()

    # replay buffer
    if all_args.share_policy:
        if all_args.use_centralized_V:
            share_obs = obs.reshape(all_args.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, axis=1)
        else:
            share_obs = obs

        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
        buffer.recurrent_hidden_states = np.zeros(
            buffer.recurrent_hidden_states.shape).astype(np.float32)
        buffer.recurrent_hidden_states_critic = np.zeros(
            buffer.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(num_agents):
            if not all_args.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            buffer[agent_id].share_obs[0] = share_obs.copy()
            buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            buffer[agent_id].recurrent_hidden_states = np.zeros(
                buffer[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            buffer[agent_id].recurrent_hidden_states_critic = np.zeros(
                buffer[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)

    # run
    start = time.time()
    episodes = int(
        all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads

    for episode in range(episodes):
        if all_args.use_linear_lr_decay:  # decrease learning rate linearly
            if all_args.share_policy:
                trainer.policy.lr_decay(episode, episodes)
            else:
                for agent_id in range(num_agents):
                    trainer[agent_id].policy.lr_decay(episode, episodes)

        for step in range(all_args.episode_length):
            # Sample actions
            with torch.no_grad():
                if all_args.share_policy:
                    trainer.prep_rollout()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                        = trainer.policy.act(torch.FloatTensor(np.concatenate(buffer.share_obs[step])),
                                             torch.FloatTensor(
                                                 np.concatenate(buffer.obs[step])),
                                             torch.FloatTensor(np.concatenate(
                                                 buffer.recurrent_hidden_states[step])),
                                             torch.FloatTensor(np.concatenate(
                                                 buffer.recurrent_hidden_states_critic[step])),
                                             torch.FloatTensor(np.concatenate(buffer.masks[step])))

                    # [envs, agents, dim]
                    values = np.array(
                        np.split(value.detach().cpu().numpy(), all_args.n_rollout_threads))
                    actions = np.array(
                        np.split(action.detach().cpu().numpy(), all_args.n_rollout_threads))
                    action_log_probs = np.array(
                        np.split(action_log_prob.detach().cpu().numpy(), all_args.n_rollout_threads))
                    recurrent_hidden_statess = np.array(np.split(
                        recurrent_hidden_states.detach().cpu().numpy(), all_args.n_rollout_threads))
                    recurrent_hidden_statess_critic = np.array(np.split(
                        recurrent_hidden_states_critic.detach().cpu().numpy(), all_args.n_rollout_threads))

                    # rearrange action
                    if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                        for i in range(envs.action_space[0].shape):
                            uc_actions_env = np.eye(
                                envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                            if i == 0:
                                actions_env = uc_actions_env
                            else:
                                actions_env = np.concatenate(
                                    (actions_env, uc_actions_env), axis=2)
                    elif envs.action_space[0].__class__.__name__ == 'Discrete':
                        actions_env = np.squeeze(
                            np.eye(envs.action_space[0].n)[actions], 2)
                    else:
                        raise NotImplementedError
                else:
                    values = []
                    actions = []
                    temp_actions_env = []
                    action_log_probs = []
                    recurrent_hidden_statess = []
                    recurrent_hidden_statess_critic = []

                    for agent_id in range(num_agents):
                        trainer[agent_id].prep_rollout()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic \
                            = trainer[agent_id].policy.act(torch.FloatTensor(buffer[agent_id].share_obs[step]),
                                                           torch.FloatTensor(
                                                               buffer[agent_id].obs[step]),
                                                           torch.FloatTensor(
                                                               buffer[agent_id].recurrent_hidden_states[step]),
                                                           torch.FloatTensor(
                                                               buffer[agent_id].recurrent_hidden_states_critic[step]),
                                                           torch.FloatTensor(buffer[agent_id].masks[step]))
                        # [agents, envs, dim]
                        values.append(value.detach().cpu().numpy())
                        action = action.detach().cpu().numpy()
                        # rearrange action
                        if envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                            for i in range(envs.action_space[agent_id].shape):
                                uc_action_env = np.eye(
                                    envs.action_space[agent_id].high[i]+1)[action[:, i]]
                                if i == 0:
                                    action_env = uc_action_env
                                else:
                                    action_env = np.concatenate(
                                        (action_env, uc_action_env), axis=1)
                        elif envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                            action_env = np.squeeze(
                                np.eye(envs.action_space[agent_id].n)[action], 1)
                        else:
                            raise NotImplementedError

                        actions.append(action)
                        temp_actions_env.append(action_env)
                        action_log_probs.append(
                            action_log_prob.detach().cpu().numpy())
                        recurrent_hidden_statess.append(
                            recurrent_hidden_states.detach().cpu().numpy())
                        recurrent_hidden_statess_critic.append(
                            recurrent_hidden_states_critic.detach().cpu().numpy())

                    # [envs, agents, dim]
                    actions_env = []
                    for i in range(all_args.n_rollout_threads):
                        one_hot_action_env = []
                        for temp_action_env in temp_actions_env:
                            one_hot_action_env.append(temp_action_env[i])
                        actions_env.append(one_hot_action_env)

                    values = np.array(values).transpose(1, 0, 2)
                    actions = np.array(actions).transpose(1, 0, 2)
                    action_log_probs = np.array(
                        action_log_probs).transpose(1, 0, 2)
                    recurrent_hidden_statess = np.array(
                        recurrent_hidden_statess).transpose(1, 0, 2)
                    recurrent_hidden_statess_critic = np.array(
                        recurrent_hidden_statess_critic).transpose(1, 0, 2)

            # Obser reward and next obs
            obs, rewards, dones, infos = envs.step(actions_env)

            # insert data in buffer
            recurrent_hidden_statess[dones == True] = np.zeros(
                ((dones == True).sum(), all_args.hidden_size)).astype(np.float32)
            recurrent_hidden_statess_critic[dones == True] = np.zeros(
                ((dones == True).sum(), all_args.hidden_size)).astype(np.float32)
            masks = np.ones((all_args.n_rollout_threads,
                             num_agents, 1)).astype(np.float32)
            masks[dones == True] = np.zeros(
                ((dones == True).sum(), 1)).astype(np.float32)

            if all_args.share_policy:
                if all_args.use_centralized_V:
                    share_obs = obs.reshape(all_args.n_rollout_threads, -1)
                    share_obs = np.expand_dims(
                        share_obs, 1).repeat(num_agents, axis=1)
                else:
                    share_obs = obs
                buffer.insert(share_obs,
                              obs,
                              recurrent_hidden_statess,
                              recurrent_hidden_statess_critic,
                              actions,
                              action_log_probs,
                              values,
                              rewards,
                              masks)
            else:
                share_obs = []
                for o in obs:
                    share_obs.append(list(itertools.chain(*o)))
                share_obs = np.array(share_obs)
                for agent_id in range(num_agents):
                    if not all_args.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    buffer[agent_id].insert(share_obs,
                                            np.array(list(obs[:, agent_id])),
                                            recurrent_hidden_statess[:,
                                                                     agent_id],
                                            recurrent_hidden_statess_critic[:,
                                                                            agent_id],
                                            actions[:, agent_id],
                                            action_log_probs[:, agent_id],
                                            values[:, agent_id],
                                            rewards[:, agent_id],
                                            masks[:, agent_id])

        if all_args.share_policy:
            # compute returns
            with torch.no_grad():
                trainer.prep_rollout()
                next_value = trainer.policy.get_value(torch.FloatTensor(np.concatenate(buffer.share_obs[-1])),
                                                      torch.FloatTensor(np.concatenate(
                                                          buffer.recurrent_hidden_states_critic[-1])),
                                                      torch.FloatTensor(np.concatenate(buffer.masks[-1])))
                next_values = np.array(
                    np.split(next_value.detach().cpu().numpy(), all_args.n_rollout_threads))
                buffer.shared_compute_returns(next_values,
                                              all_args.use_gae,
                                              all_args.gamma,
                                              all_args.gae_lambda,
                                              all_args.use_proper_time_limits,
                                              all_args.use_popart,
                                              trainer.value_normalizer)
            # update network
            trainer.prep_training()
            value_loss, actor_grad_norm, action_loss, dist_entropy, critic_grad_norm, joint_loss, joint_grad_norm = trainer.shared_update(
                buffer)
            # clean the buffer and reset
            buffer.after_update()
        else:
            value_losses = []
            action_losses = []
            dist_entropies = []
            actor_grad_norms = []
            critic_grad_norms = []
            joint_losses = []
            joint_grad_norms = []
            for agent_id in range(num_agents):
                # compute returns
                with torch.no_grad():
                    trainer[agent_id].prep_rollout()
                    next_value = trainer[agent_id].policy.get_value(torch.FloatTensor(buffer[agent_id].share_obs[-1]),
                                                                    torch.FloatTensor(
                                                                        buffer[agent_id].recurrent_hidden_states_critic[-1]),
                                                                    torch.FloatTensor(buffer[agent_id].masks[-1]))
                    next_value = next_value.detach().cpu().numpy()
                    buffer[agent_id].compute_returns(next_value,
                                                     all_args.use_gae,
                                                     all_args.gamma,
                                                     all_args.gae_lambda,
                                                     all_args.use_proper_time_limits,
                                                     all_args.use_popart,
                                                     trainer[agent_id].value_normalizer)
                # update network
                trainer[agent_id].prep_training()
                value_loss, critic_grad_norm, action_loss, dist_entropy, actor_grad_norm, joint_loss, joint_grad_norm = trainer[agent_id].separated_update(
                    agent_id, buffer[agent_id])

                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                actor_grad_norms.append(actor_grad_norm)
                critic_grad_norms.append(critic_grad_norm)
                joint_losses.append(joint_loss)
                joint_grad_norms.append(joint_grad_norm)

                buffer[agent_id].after_update()

        # post process
        total_num_steps = (episode + 1) * \
            all_args.episode_length * all_args.n_rollout_threads

        # save model
        # save for every interval-th episode or for the last epoch
        if (episode % all_args.save_interval == 0 or episode == episodes - 1):
            if all_args.share_policy:
                torch.save({
                    'model': trainer.policy
                },
                    str(run_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):
                    torch.save({
                        'model': trainer[agent_id].policy
                    },
                        str(run_dir) + "/agent%i_model" % agent_id + ".pt")

        # log information
        if episode % all_args.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                  .format(all_args.scenario_name,
                          all_args.algorithm_name,
                          all_args.experiment_name,
                          episode,
                          episodes,
                          total_num_steps,
                          all_args.num_env_steps,
                          int(total_num_steps / (end - start))))
            if all_args.share_policy:
                print("value loss of agent: " + str(value_loss))
                print("average episode rewards of agent: " +
                      str(np.mean(buffer.rewards) * all_args.episode_length))
                if all_args.use_wandb:
                    wandb.log({"value_loss": value_loss}, step=total_num_steps)
                    wandb.log({"action_loss": action_loss},
                              step=total_num_steps)
                    wandb.log({"dist_entropy": dist_entropy},
                              step=total_num_steps)
                    wandb.log({"actor_grad_norm": actor_grad_norm},
                              step=total_num_steps)
                    wandb.log({"critic_grad_norm": critic_grad_norm},
                              step=total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        wandb.log({"joint_loss": joint_loss}, step=total_num_steps)
                        wandb.log({"joint_grad_norm": joint_grad_norm}, step=total_num_steps)
                    wandb.log({"average_episode_rewards": np.mean(
                        buffer.rewards) * all_args.episode_length}, step=total_num_steps)
                else:
                    writter.add_scalars(
                        "value_loss", {"value_loss": value_loss}, total_num_steps)
                    writter.add_scalars(
                        "action_loss", {"action_loss": action_loss}, total_num_steps)
                    writter.add_scalars(
                        "dist_entropy", {"dist_entropy": dist_entropy}, total_num_steps)
                    writter.add_scalars(
                        "actor_grad_norm", {"actor_grad_norm": actor_grad_norm}, total_num_steps)
                    writter.add_scalars(
                        "critic_grad_norm", {"critic_grad_norm": critic_grad_norm}, total_num_steps)
                    if "mappg" in all_args.algorithm_name:
                        writter.add_scalars(
                            "joint_loss", {"joint_loss": joint_loss}, total_num_steps)
                        writter.add_scalars(
                            "joint_grad_norm", {"joint_grad_norm": joint_grad_norm}, total_num_steps)
                    writter.add_scalars("average_episode_rewards", {"average_episode_rewards": np.mean(
                        buffer.rewards) * all_args.episode_length}, total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("value loss of agent%i: " %
                          agent_id + str(value_losses[agent_id]))
                    print("average episode rewards of agent%i: " % agent_id +
                          str(np.mean(buffer[agent_id].rewards) * all_args.episode_length))
                    if all_args.use_wandb:
                        wandb.log(
                            {"agent%i/value_loss" % agent_id: value_losses[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/action_loss" % agent_id: action_losses[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, step=total_num_steps)
                        wandb.log(
                            {"agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, step=total_num_steps)
                        if "mappg" in all_args.algorithm_name:   
                            wandb.log(
                                {"agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, step=total_num_steps)
                            wandb.log(
                                {"agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, step=total_num_steps)
                        wandb.log({"agent%i/average_episode_rewards" % agent_id: np.mean(
                            buffer[agent_id].rewards) * all_args.episode_length}, step=total_num_steps)
                    else:
                        writter.add_scalars("agent%i/value_loss" % agent_id, {
                                            "agent%i/value_loss" % agent_id: value_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/action_loss" % agent_id, {
                                            "agent%i/action_loss" % agent_id: action_losses[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/dist_entropy" % agent_id, {
                                            "agent%i/dist_entropy" % agent_id: dist_entropies[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/actor_grad_norm" % agent_id, {
                                            "agent%i/actor_grad_norm" % agent_id: actor_grad_norms[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/critic_grad_norm" % agent_id, {
                                            "agent%i/critic_grad_norm" % agent_id: critic_grad_norms[agent_id]}, total_num_steps)
                        if "mappg" in all_args.algorithm_name: 
                            writter.add_scalars("agent%i/joint_loss" % agent_id, {
                                                "agent%i/joint_loss" % agent_id: joint_losses[agent_id]}, total_num_steps)
                            writter.add_scalars("agent%i/joint_grad_norm" % agent_id, {
                                                "agent%i/joint_grad_norm" % agent_id: joint_grad_norms[agent_id]}, total_num_steps)
                        writter.add_scalars("agent%i/average_episode_rewards" % agent_id, {"agent%i/average_episode_rewards" % agent_id: np.mean(
                            buffer[agent_id].rewards) * all_args.episode_length}, total_num_steps)

            if all_args.env_name == "MPE":
                for agent_id in range(num_agents):
                    show_rewards = []
                    for info in infos:
                        if 'individual_reward' in info[agent_id].keys():
                            show_rewards.append(
                                info[agent_id]['individual_reward'])
                    if all_args.use_wandb:
                        wandb.log({'agent%i/individual_rewards' %
                                   agent_id: np.mean(show_rewards)}, step=total_num_steps)
                    else:
                        writter.add_scalars('agent%i/individual_rewards' % agent_id, {
                                            'agent%i/individual_rewards' % agent_id: np.mean(show_rewards)}, total_num_steps)
        if episode % all_args.eval_interval == 0 and all_args.eval:
            eval_episode_rewards = []
            eval_obs = eval_envs.reset()

            if all_args.share_policy:
                eval_share_obs = eval_obs.reshape(
                    all_args.n_eval_rollout_threads, -1)
                eval_share_obs = np.expand_dims(
                    eval_share_obs, 1).repeat(num_agents, axis=1)
            else:
                eval_share_obs = []
                for o in eval_obs:
                    eval_share_obs.append(list(itertools.chain(*o)))
                eval_share_obs = np.array(eval_share_obs)
            eval_recurrent_hidden_states = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_recurrent_hidden_states_critic = np.zeros(
                (all_args.n_eval_rollout_threads, num_agents, all_args.hidden_size)).astype(np.float32)
            eval_masks = np.ones(
                (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)

            for eval_step in range(all_args.episode_length):
                if all_args.share_policy:
                    if not all_args.use_centralized_V:
                        eval_share_obs = eval_obs
                    trainer.prep_rollout()
                    _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer.policy.act(torch.FloatTensor(np.concatenate(eval_share_obs)),
                                                                                                                            torch.FloatTensor(np.concatenate(eval_obs)),
                                                                                                                            torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states)),
                                                                                                                            torch.FloatTensor(np.concatenate(eval_recurrent_hidden_states_critic)),
                                                                                                                            torch.FloatTensor(np.concatenate(eval_masks)),
                                                                                                                            deterministic=True)
                    eval_actions = np.array(
                        np.split(eval_action.detach().cpu().numpy(), all_args.n_eval_rollout_threads))
                    eval_recurrent_hidden_states = np.array(np.split(
                        eval_recurrent_hidden_state.detach().cpu().numpy(), all_args.n_eval_rollout_threads))
                    eval_recurrent_hidden_states_critic = np.array(np.split(
                        eval_recurrent_hidden_state_critic.detach().cpu().numpy(), all_args.n_eval_rollout_threads))

                    if eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                        for i in range(eval_envs.action_space[0].shape):
                            eval_uc_actions_env = np.eye(
                                eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                            if i == 0:
                                eval_actions_env = eval_uc_actions_env
                            else:
                                eval_actions_env = np.concatenate(
                                    (eval_actions_env, eval_uc_actions_env), axis=2)
                    elif eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                        eval_actions_env = np.squeeze(
                            np.eye(eval_envs.action_space[0].n)[eval_actions], 2)
                    else:
                        raise NotImplementedError
                else:
                    eval_temp_actions_env = []
                    for agent_id in range(num_agents):
                        if not all_args.use_centralized_V:
                            eval_share_obs = np.array(
                                list(eval_obs[:, agent_id]))
                        trainer[agent_id].prep_rollout()
                        _, eval_action, _, eval_recurrent_hidden_state, eval_recurrent_hidden_state_critic = trainer[agent_id].policy.act(torch.FloatTensor(eval_share_obs),
                                                                                                                                          torch.FloatTensor(np.array(list(eval_obs[:, agent_id]))),
                                                                                                                                        torch.FloatTensor(eval_recurrent_hidden_states[:, agent_id]),
                                                                                                                                        torch.FloatTensor(eval_recurrent_hidden_states_critic[:, agent_id]),
                                                                                                                                        torch.FloatTensor(eval_masks[:, agent_id]),
                                                                                                                                        deterministic=True)

                        eval_action = eval_action.detach().cpu().numpy()
                        # rearrange action
                        if eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                            for i in range(eval_envs.action_space[agent_id].shape):
                                eval_uc_action_env = np.eye(
                                    eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                                if i == 0:
                                    eval_action_env = eval_uc_action_env
                                else:
                                    eval_action_env = np.concatenate(
                                        (eval_action_env, eval_uc_action_env), axis=1)
                        elif eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                            eval_action_env = np.squeeze(
                                np.eye(eval_envs.action_space[agent_id].n)[eval_action], 1)
                        else:
                            raise NotImplementedError
                        eval_temp_actions_env.append(eval_action_env)
                        eval_recurrent_hidden_states[:, agent_id] = eval_recurrent_hidden_state.detach(
                        ).cpu().numpy()
                        eval_recurrent_hidden_states_critic[:, agent_id] = eval_recurrent_hidden_state_critic.detach(
                        ).cpu().numpy()

                    # [envs, agents, dim]
                    eval_actions_env = []
                    for i in range(all_args.n_eval_rollout_threads):
                        eval_one_hot_action_env = []
                        for eval_temp_action_env in eval_temp_actions_env:
                            eval_one_hot_action_env.append(
                                eval_temp_action_env[i])
                        eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(
                    eval_actions_env)
                eval_episode_rewards.append(eval_rewards)
                if all_args.share_policy:
                    eval_share_obs = eval_obs.reshape(
                        all_args.n_eval_rollout_threads, -1)
                    eval_share_obs = np.expand_dims(
                        eval_share_obs, 1).repeat(num_agents, axis=1)
                else:
                    eval_share_obs = []
                    for o in eval_obs:
                        eval_share_obs.append(list(itertools.chain(*o)))
                    eval_share_obs = np.array(eval_share_obs)

                eval_recurrent_hidden_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), all_args.hidden_size)).astype(np.float32)
                eval_recurrent_hidden_states_critic[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), all_args.hidden_size)).astype(np.float32)
                eval_masks = np.ones(
                    (all_args.n_eval_rollout_threads, num_agents, 1)).astype(np.float32)
                eval_masks[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), 1)).astype(np.float32)

            eval_episode_rewards = np.array(eval_episode_rewards)
            if all_args.share_policy:
                print("eval average episode rewards of agent: " +
                      str(np.mean(np.sum(eval_episode_rewards, axis=0))))
                if all_args.use_wandb:
                    wandb.log({"eval_average_episode_rewards": np.mean(
                        np.sum(eval_episode_rewards, axis=0))}, step=total_num_steps)
                else:
                    writter.add_scalars("eval_average_episode_rewards", {"eval_average_episode_rewards": np.mean(
                        np.sum(eval_episode_rewards, axis=0))}, total_num_steps)
            else:
                for agent_id in range(num_agents):
                    print("eval average episode rewards of agent%i: " % agent_id +
                          str(np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))))
                    if all_args.use_wandb:
                        wandb.log({"agent%i/eval_average_episode_rewards" % agent_id: np.mean(
                            np.sum(eval_episode_rewards[:, :, agent_id], axis=0))}, step=total_num_steps)
                    else:
                        writter.add_scalars("agent%i/eval_average_episode_rewards" % agent_id, {"agent%i/eval_average_episode_rewards" % agent_id: np.mean(
                            np.sum(eval_episode_rewards[:, :, agent_id], axis=0))}, total_num_steps)
    envs.close()
    if all_args.eval:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        writter.export_scalars_to_json(str(log_dir + '/summary.json'))
        writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])