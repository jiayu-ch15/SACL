    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class AgarRunner(Runner):
    def __init__(self, config):
        super(AgarRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, \
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
                
                if self.env_name == "Agar":
                    env_infos = []
                    for agent_id in range(self.num_agents):
                        env_info = {}

                        env_info['collective_return'] = []
                        env_info['split'] = []
                        env_info['hunt'] = []
                        env_info['attack'] = []
                        env_info['cooperate'] = []
                        env_info['original_return'] = []

                        for info in infos:                    
                            if 'collective_return' in info[agent_id].keys():
                                env_info['collective_return'].append(info[agent_id]['collective_return']) 
                            if 'behavior' in info[agent_id].keys():
                                env_info['split'].append(info[agent_id]['behavior'][0])
                                env_info['hunt'].append(info[agent_id]['behavior'][1])
                                env_info['attack'].append(info[agent_id]['behavior'][2])
                                env_info['cooperate'].append(info[agent_id]['behavior'][3])
                            if 'o_r' in info[agent_id].keys():
                                env_info['original_return'].append(info[agent_id]['o_r']) 
                        
                        env_infos.append(env_info)

                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards)*self.episode_length})
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

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
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3) #[rollout_threads, agents,...]
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
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
                                        active_masks[:, agent_id])

    def log_env(self, env_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in env_infos[agent_id].items():
                if len(v)>0:
                    agent_k = "agent%i/" % agent_id + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(v)}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(v)}, total_num_steps)

    def log_eval(self, eval_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in eval_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)     

    @torch.no_grad()
    def render(self):
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                for i in range(self.num_agents):
                    image = np.squeeze(envs.render(mode="rgb_array", playeridx=i))
                    all_frames.append(image)
                    #time.sleep(self.all_args.ifi)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    temp_actions_env.append(action)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # Obser reward and next obs
                actions = np.transpose(np.array(temp_actions_env),(1,0,2))
                obs, rewards, dones, infos = self.envs.step(actions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    for i in range(self.num_agents):
                        image = np.squeeze(envs.render(mode="rgb_array", playeridx=i))
                        all_frames.append(image)
                        #time.sleep(self.all_args.ifi)

                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            render_infos = []
            print("#########episode no."+str(episode))
            for agent_id in range(self.num_agents):
                env_info = {}

                env_info['collective_return'] = []
                env_info['split'] = []
                env_info['hunt'] = []
                env_info['attack'] = []
                env_info['cooperate'] = []
                env_info['original_return'] = []

                for info in infos:                    
                    if 'collective_return' in info[agent_id].keys():
                        env_info['collective_return'].append(info[agent_id]['collective_return']) 
                    if 'behavior' in info[agent_id].keys():
                        env_info['split'].append(info[agent_id]['behavior'][0])
                        env_info['hunt'].append(info[agent_id]['behavior'][1])
                        env_info['attack'].append(info[agent_id]['behavior'][2])
                        env_info['cooperate'].append(info[agent_id]['behavior'][3])
                    if 'o_r' in info[agent_id].keys():
                        env_info['original_return'].append(info[agent_id]['o_r']) 
                        
                env_info['average_episode_reward'] = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("agent no."+str(agent_id))
                print(env_info)
                render_infos.append(env_info)


            if self.all_args.save_gifs:
                for i in range(self.num_agents):
                    imageio.mimsave(str(self.gif_dir) + '/render_agent' + str(i) + '_episode' + str(episode) + '.gif', all_frames[i+self.episode_length*episode:i+self.episode_length*(episode+1):self.num_agents], 'GIF', duration=self.all_args.ifi)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        for i in range(self.num_agents):
            eval_episode_rewards.append([])
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                            eval_rnn_states[:, agent_id],
                                                                            eval_masks[:, agent_id],
                                                                            deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                
                eval_temp_actions_env.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            
            for i in range(self.num_agents):
                temp = eval_rewards[:, i, :]
                eval_episode_rewards[i].append(np.mean(temp))           
            
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        #eval_episode_rewards = np.array(eval_episode_rewards) 
        eval_train_infos = []
        for i in range(self.num_agents):
            eval_train_infos.append({})

        for agent_id in range(self.num_agents):
            eval_train_infos[agent_id]['eval_average_episode_rewards'] = np.mean(eval_episode_rewards[agent_id])*self.episode_length
            #print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))


        self.log_eval(eval_train_infos, total_num_steps)  