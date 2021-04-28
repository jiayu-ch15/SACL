    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio
from icecream import ic
from collections import defaultdict, deque

from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class AgarRunner(Runner):
    def __init__(self, config):
        super(AgarRunner, self).__init__(config)
        self.env_infos = [defaultdict(list) for _ in range(self.num_agents)]

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

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

                for agent_id in range(self.num_agents):                        
                    train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = [defaultdict(list) for _ in range(self.num_agents)]

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
       
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        dones_env = np.all(dones, axis=-1)
        for done, info in zip(dones_env, infos):
            if done:
                for agent_id in range(self.num_agents):
                    if 'collective_return' in info[agent_id].keys():
                        self.env_infos[agent_id]['collective_return'].append(info[agent_id]['collective_return']) 
                    if 'behavior' in info[agent_id].keys():
                        self.env_infos[agent_id]['split'].append(info[agent_id]['behavior'][0])
                        self.env_infos[agent_id]['hunt'].append(info[agent_id]['behavior'][1])
                        self.env_infos[agent_id]['attack'].append(info[agent_id]['behavior'][2])
                        self.env_infos[agent_id]['cooperate'].append(info[agent_id]['behavior'][3])
                    if 'o_r' in info[agent_id].keys():
                        self.env_infos[agent_id]['original_return'].append(info[agent_id]['o_r'])

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
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

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks)
    
    def log_env(self, env_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in env_infos[agent_id].items():
                if len(v) > 0:
                    agent_k = "agent%i/" % agent_id + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(v)}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(v)}, total_num_steps)
    
    @torch.no_grad()
    def render(self):
        envs = self.envs
        f = str(self.run_dir / 'log.txt')

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            end = False
            step = 0
            episode_rewards = []

            obs = envs.reset()
            if self.all_args.save_gifs:
                for i in range(self.num_agents):
                    image = np.squeeze(envs.render(mode="rgb_array", playeridx=i))
                    all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            while not end:
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions)
                episode_rewards.append(rewards)

                dones_env = np.all(dones, axis=1)
                end = dones_env[0]

                rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    for i in range(self.num_agents):
                        image = np.squeeze(envs.render(mode="rgb_array", playeridx=i))
                        all_frames.append(image)

                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            render_infos = []
            with open(f,'a') as file:
                file.write("\n########## episode no." + str(episode) + ' ##########\n')
            print("\n########## episode no." + str(episode) + ' ##########')
            print("\nrender step is {}.\n".format(step))
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
                        
                env_info['average_episode_reward'] = np.mean(np.sum(np.array(episode_rewards), axis=0))
                with open(f,'a') as file:
                    file.write("\n@@@@ agent no." + str(agent_id) + ' @@@@\n')
                print("\n@@@@ agent no." + str(agent_id) + ' @@@@')
                with open(f, 'a') as file:
                    for k,v in env_info.items():
                        file.write('\n' + str(k) + ' : ' + str(v) + '\n')
                print(env_info)
                render_infos.append(env_info)


            if self.all_args.save_gifs:
                for i in range(self.num_agents):
                    imageio.mimsave(str(self.gif_dir) + '/render_agent' + str(i) + '_episode' + str(episode) + '.gif', all_frames[i+self.episode_length*episode:i+self.episode_length*(episode+1):self.num_agents], 'GIF', duration=self.all_args.ifi)
            
    @torch.no_grad()
    def eval(self, total_num_steps):

        action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        eval_env_infos = [defaultdict(list) for _ in range(self.num_agents)]
        eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0

        eval_obs = self.eval_envs.reset(eval_reset_choose)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones_env = np.zeros(self.n_eval_rollout_threads, dtype=bool)

        while True:
            eval_choose = (eval_dones_env==False)

            if ~np.any(eval_choose):
                break
            eval_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, action_shape)).astype(np.int) * (-1)
            
            self.trainer.prep_rollout()
            eval_action, eval_rnn_state = self.trainer.policy.act(np.concatenate(eval_obs[eval_choose]),
                                                np.concatenate(eval_rnn_states[eval_choose]),
                                                np.concatenate(eval_masks[eval_choose]),
                                                deterministic=True)

            eval_actions[eval_choose] = np.array(np.split(_t2n(eval_action), (eval_choose == True).sum()))
            eval_rnn_states[eval_choose] = np.array(np.split(_t2n(eval_rnn_state), (eval_choose == True).sum()))
            
            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions) 
            eval_dones_env = np.all(eval_dones, axis=-1)

            for eval_done, eval_info in zip(eval_dones_env, eval_infos):
                if eval_done:
                    for agent_id in range(self.num_agents):
                        if 'collective_return' in eval_info[agent_id].keys():
                            eval_env_infos[agent_id]['eval_collective_return'].append(eval_info[agent_id]['collective_return']) 
                        if 'behavior' in eval_info[agent_id].keys():
                            eval_env_infos[agent_id]['eval_split'].append(eval_info[agent_id]['behavior'][0])
                            eval_env_infos[agent_id]['eval_hunt'].append(eval_info[agent_id]['behavior'][1])
                            eval_env_infos[agent_id]['eval_attack'].append(eval_info[agent_id]['behavior'][2])
                            eval_env_infos[agent_id]['eval_cooperate'].append(eval_info[agent_id]['behavior'][3])
                        if 'o_r' in eval_info[agent_id].keys():
                            eval_env_infos[agent_id]['eval_original_return'].append(eval_info[agent_id]['o_r'])

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.log_env(eval_env_infos, total_num_steps)