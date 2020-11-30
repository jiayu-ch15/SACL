    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if "mappo" in self.algorithm_name:
            if self.use_single_network:
                from onpolicy.algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif "mappg" in self.algorithm_name:
            if self.use_single_network:
                from onpolicy.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                from onpolicy.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
                from onpolicy.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            raise NotImplementedError

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            if self.model_dir == None or self.model_dir == "":
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device = self.device,
                            cat_self = False if self.use_obs_instead_of_state else True)
            else:
                po = torch.load(str(self.model_dir) + "/agent" + str(agent_id) + "_model.pt")['model']
            # algorithm
            tr = TrainAlgo(self.all_args, po, device = self.device)
            # buffer
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            env_infos = {}

            env_infos['max_box_move_prep'] = []
            env_infos['max_box_move'] = []
            env_infos['num_box_lock_prep'] = []
            env_infos['num_box_lock'] = []
            env_infos['max_ramp_move_prep'] = []
            env_infos['max_ramp_move'] = []
            env_infos['num_ramp_lock_prep'] = []
            env_infos['num_ramp_lock'] = []
            env_infos['food_eaten_prep'] = []
            env_infos['food_eaten'] = []
            env_infos['lock_rate'] = []
            env_infos['activated_sites'] = [] 

            discard_episode = 0
            success = 0
            trials = 0

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)

                for done, info in zip(dones, infos):
                    if done:
                        if "discard_episode" in info.keys() and info['discard_episode']:
                            discard_episode += 1
                        else:
                            trials += 1

                        if "success" in info.keys() and info['success']:
                            success += 1

                        for k in env_infos.keys():
                            if k in info.keys():
                                env_infos[k].append(info[k])

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "HideAndSeek" :                                              
                    for hider_id in range(self.all_args.num_hiders):
                        agent_k = 'hider%i/average_step_rewards' % hider_id
                        env_infos[agent_k] = list(self.buffer[hider_id].rewards)
                    for seeker_id in range(self.all_args.num_seekers):
                        agent_k = 'seeker%i/average_step_rewards' % seeker_id
                        env_infos[agent_k] = list(self.buffer[self.all_args.num_hiders+seeker_id].rewards)

                if self.env_name == "BoxLocking" or self.env_name == "BlueprintConstruction":
                    for agent_id in range(self.num_agents):
                        train_info = {'average_step_rewards': np.mean(self.buffer[agent_id].rewards)}
                        train_infos.append(train_info)

                    success_rate = success/trials if trials > 0 else 0.0
                    print("success rate is {}.".format(success_rate))
                    if self.use_wandb:
                        wandb.log({'success_rate': success_rate}, step=total_num_steps)
                        wandb.log({'discard_episode': discard_episode}, step=total_num_steps)
                    else:
                        self.writter.add_scalars('success_rate', {'success_rate': success_rate}, total_num_steps)
                        self.writter.add_scalars('discard_episode', {'discard_episode': discard_episode}, total_num_steps)

                self.log_env(env_infos, total_num_steps)
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        share_obs = share_obs if self.use_centralized_V else obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):    
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])

            values.append(_t2n(value))
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        share_obs = share_obs if self.use_centralized_V else obs
        if len(rewards.shape) < 3:
            rewards = rewards[:, :, np.newaxis]

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], 
                                        obs[:, agent_id], 
                                        rnn_states[:, agent_id], 
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id], 
                                        action_log_probs[:, agent_id], 
                                        values[:, agent_id], 
                                        rewards[:, agent_id], 
                                        masks[:, agent_id])

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            torch.save({'model': self.trainer[agent_id].policy}, self.save_dir + "/agent%i_model" % agent_id + ".pt")
    
    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_envs = self.eval_envs
        action_shape = eval_envs.action_space[0].shape

        eval_env_infos = {}

        eval_env_infos['eval_num_box_lock_prep'] = []
        eval_env_infos['eval_num_box_lock'] = []
        eval_env_infos['eval_activated_sites'] = []
        eval_env_infos['eval_lock_rate'] = []

        eval_success = 0
        eval_trials = 0

        eval_episode_rewards = 0

        eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0

        eval_obs, eval_share_obs, _ = eval_envs.reset(eval_reset_choose)

        eval_share_obs = eval_share_obs if self.use_centralized_V else eval_obs

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.hidden_size), dtype=np.float32)
        eval_rnn_states_critic = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones = np.zeros(self.n_eval_rollout_threads, dtype=bool)

        while True:
            eval_choose = eval_dones == False
            if ~np.any(eval_choose):
                break
            with torch.no_grad():
                eval_actions = []
                for agent_id in range(self.num_agents):
                    agent_eval_actions = np.ones((self.n_eval_rollout_threads, action_shape)).astype(np.int) * (-1)
                    self.trainer[agent_id].prep_rollout()
                    _, eval_action, _, eval_rnn_state, eval_rnn_state_critic \
                        = self.trainer[agent_id].policy.get_actions(eval_share_obs[eval_choose, agent_id],
                                                                    eval_obs[eval_choose, agent_id],
                                                                    eval_rnn_states[eval_choose, agent_id],          
                                                                    eval_rnn_states_critic[eval_choose, agent_id],
                                                                    eval_masks[eval_choose, agent_id],
                                                                    deterministic=True)
                    agent_eval_actions[eval_choose] = _t2n(eval_action)
                    eval_actions.append(agent_eval_actions)
                    eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)
                    eval_rnn_states_critic[eval_choose, agent_id] = _t2n(eval_rnn_state_critic)

                eval_actions = np.array(eval_actions).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = eval_envs.step(eval_actions)
            eval_share_obs = eval_share_obs if self.use_centralized_V else eval_obs

            eval_episode_rewards += eval_rewards

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_rnn_states_critic[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, 1), dtype=np.float32)

            discard_reset_choose = np.zeros(self.n_eval_rollout_threads, dtype=bool)

            for eval_done, eval_info, discard in zip(eval_dones, eval_infos, discard_reset_choose):
                if eval_done:
                    if "discard_episode" in eval_info.keys() and eval_info['discard_episode']:
                        discard = True
                    else:
                        eval_trials += 1

                    if "success" in eval_info.keys() and eval_info['success']:
                        eval_success += 1
                    
                    for k in eval_env_infos.keys():
                        if k in eval_info.keys():
                            eval_env_infos[k].append(eval_info[k])                   

            discard_obs, discard_share_obs, _ = eval_envs.reset(discard_reset_choose)

            eval_obs[discard_reset_choose == True] = discard_obs[discard_reset_choose == True]
            eval_share_obs[discard_reset_choose == True] = discard_share_obs[discard_reset_choose == True]
            eval_dones[discard_reset_choose == True] = np.zeros(discard_reset_choose.sum(), dtype=bool)

        if self.env_name == "BoxLocking" or self.env_name == "BlueprintConstruction":
            eval_success_rate = eval_success/eval_trials if eval_trials > 0 else 0.0
            print("eval success rate is {}.".format(eval_success_rate))
            if self.use_wandb:
                wandb.log({'eval_success_rate': eval_success_rate}, step=total_num_steps)
            else:
                self.writter.add_scalars('eval_success_rate', {'eval_success_rate': eval_success_rate}, total_num_steps)
            
        self.log_env(eval_env_infos, total_num_steps)