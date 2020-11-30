    
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

        last_battles_game = np.zeros(self.n_rollout_threads)
        last_battles_won = np.zeros(self.n_rollout_threads)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
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
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                
                for agent_id in range(self.num_agents):
                    train_infos.append({"average_step_rewards": np.mean(self.buffer[agent_id].rewards)})
            
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  
        else:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

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
                                                            self.buffer[agent_id].masks[step],
                                                            self.buffer[agent_id].available_actions[step])

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
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if self.use_centralized_V:
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], 
                                        obs[:, agent_id], 
                                        rnn_states[:, agent_id], 
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id], 
                                        action_log_probs[:, agent_id], 
                                        values[:, agent_id], 
                                        rewards[:, agent_id], 
                                        masks[:, agent_id], 
                                        bad_masks[:, agent_id], 
                                        active_masks[:, agent_id], 
                                        available_actions[:, agent_id])

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
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        if self.use_centralized_V:
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            eval_share_obs = eval_obs

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.hidden_size), dtype=np.float32)
        eval_rnn_states_critic = np.zeros_like(eval_rnn_states)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                _, eval_action, _, eval_rnn_state, eval_rnn_state_critic = \
                    self.trainer[agent_id].policy.get_actions(eval_share_obs[:, agent_id],
                                                            eval_obs[:, agent_id],
                                                            eval_rnn_states[:, agent_id],
                                                            eval_rnn_states_critic[:, agent_id],
                                                            eval_masks[:, agent_id],
                                                            eval_available_actions[:, agent_id],
                                                            deterministic=True)

                eval_actions.append(_t2n(eval_action))
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                eval_rnn_states_critic[:, agent_id] = _t2n(eval_rnn_state_critic)

            eval_actions = np.array(eval_actions).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            
            if self.use_centralized_V:
                eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                eval_share_obs = eval_obs

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_rnn_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break