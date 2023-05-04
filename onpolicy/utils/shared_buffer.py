import torch
import numpy as np
from collections import defaultdict

from onpolicy.utils.util import check,get_shape_from_obs_space, get_shape_from_act_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

class SharedReplayBuffer_football_ensemble(object):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.num_critic = args.num_critic
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits 

        self._mixed_obs = False  # for mixed observation   

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # for mixed observation
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}
            self.share_obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
            for key in share_obs_shape:
                self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape[key].shape), dtype=np.float32)
        
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if type(share_obs_shape[-1]) == list:
                share_obs_shape = share_obs_shape[:1]

            self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_critic, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_critic), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
               
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):

        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][self.step + 1] = share_obs[key].copy()
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.share_obs[self.step + 1] = share_obs.copy()
            self.obs[self.step + 1] = obs.copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][0] = self.share_obs[key][-1].copy()
            for key in self.obs.keys():
                self.obs[key][0] = self.obs[key][-1].copy()
        else:
            self.share_obs[0] = self.share_obs[-1].copy()
            self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1] 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                # change for ensemble
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def feed_forward_generator_ensemble(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, _, num_agents = self.rewards.shape[0:3]
        n_rollout_threads = self.n_rollout_threads
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        self.shuffle_batch = True
        if self.shuffle_batch:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

            # diff mini_batch for multi-critic
            rand_critic = []
            sampler_critic = []
            for critic_id in range(self.num_critic):
                rand_critic.append(torch.randperm(batch_size).numpy())
                sampler_critic.append([rand_critic[critic_id][i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)])
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()
        else:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
            sampler_critic = []
            for critic_id in range(self.num_critic):
                sampler_critic.append(sampler_actor)
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            obs_actor = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states_actor = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        actions_actor = self.actions[:].reshape(-1, self.actions.shape[-1])
        masks_actor = self.masks[:-1].reshape(-1, 1)
        active_masks_actor = self.active_masks[:-1].reshape(-1, 1)
        advantages_actor = advantages.reshape(-1, 1)
        action_log_probs_actor = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        
        share_obs_critic = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        masks_critic = self.masks[:-1].reshape(-1, 1)
        value_preds_critic = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[3:])
        returns_critic = self.returns[:-1].reshape(-1, *self.returns.shape[3:])
        active_masks_critic = self.active_masks[:-1].reshape(-1, 1)

        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        
        for indices_actor,indices_critic in zip(sampler_actor, sampler_critic):
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            # indices_critic : num_critic * index
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                obs_actor_batch = obs_actor[indices_actor]
            rnn_states_actor_batch = rnn_states_actor[indices_actor]
            actions_actor_batch = actions_actor[indices_actor]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices_actor]
            else:
                available_actions_batch = None
            masks_actor_batch = masks_actor[indices_actor]
            active_masks_actor_batch = active_masks_actor[indices_actor]
            old_action_log_probs_actor_batch = action_log_probs_actor[indices_actor]

            # diff critic batch
            share_obs_critic_batch = []
            rnn_states_critic_batch = []
            value_preds_critic_batch = []
            return_critic_batch = []
            masks_critic_batch = []
            active_masks_critic_batch = []
            for critic_id in range(self.num_critic):
                share_obs_critic_batch.append(share_obs_critic[indices_critic[critic_id]])
                # TODO invalid rnn_states_critic_batch
                rnn_states_critic_batch.append(rnn_states_critic[indices_critic[critic_id]])
                value_preds_critic_batch.append(value_preds_critic[indices_critic[critic_id],critic_id])
                return_critic_batch.append(returns_critic[indices_critic[critic_id],critic_id])
                masks_critic_batch.append(masks_critic[indices_critic[critic_id]])
                active_masks_critic_batch.append(active_masks_critic[indices_critic[critic_id]])
            share_obs_critic_batch = np.stack(share_obs_critic_batch,axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch,axis=1)
            value_preds_critic_batch = np.stack(value_preds_critic_batch, axis=1)
            return_critic_batch = np.stack(return_critic_batch, axis=1)
            masks_critic_batch = np.stack(masks_critic_batch, axis=1)
            active_masks_critic_batch = np.stack(active_masks_critic_batch, axis=1)

            if advantages is None:
                adv_targ_actor = None
            else:
                adv_targ_actor = advantages_actor[indices_actor]

            yield obs_actor_batch, rnn_states_actor_batch, actions_actor_batch, masks_actor_batch, active_masks_actor_batch, adv_targ_actor, old_action_log_probs_actor_batch, available_actions_batch, \
                  share_obs_critic_batch, rnn_states_critic_batch, masks_critic_batch, value_preds_critic_batch, return_critic_batch, active_masks_critic_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads*num_agents
        assert n_rollout_threads*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()
        
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key].reshape(-1, batch_size, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key].reshape(-1, batch_size, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
            obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][:-1, ind])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][:-1, ind])
                else:
                    share_obs_batch.append(share_obs[:-1, ind])
                    obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
            
            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = np.stack(share_obs_batch[key], 1)
                for key in obs_batch.keys():
                    obs_batch[key] = np.stack(obs_batch[key], 1)
            else:
                share_obs_batch = np.stack(share_obs_batch, 1)
                obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = _flatten(T, N, share_obs_batch[key])
                for key in obs_batch.keys():
                    obs_batch[key] = _flatten(T, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(T, N, share_obs_batch)
                obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                if len(self.share_obs[key].shape) == 6:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs[key].shape[3:])
                elif len(self.share_obs[key].shape) == 5:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.share_obs[key].shape[3:])
                else:
                    share_obs[key] = _cast(self.share_obs[key][:-1])
                   
            for key in self.obs.keys():
                if len(self.obs[key].shape) == 6:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs[key].shape[3:])
                elif len(self.obs[key].shape) == 5:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.obs[key].shape[3:])
                else:
                    obs[key] = _cast(self.obs[key][:-1])
        else:
            if len(self.share_obs.shape) > 4:
                share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
                obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            else:
                share_obs = _cast(self.share_obs[:-1])
                obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])       
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                else:
                    share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                    obs_batch.append(obs[ind:ind+data_chunk_length])

                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim) 
            if self._mixed_obs:
                for key in share_obs_batch.keys():  
                    share_obs_batch[key] = np.stack(share_obs_batch[key], axis=1)
                for key in obs_batch.keys():  
                    obs_batch[key] = np.stack(obs_batch[key], axis=1)
            else:        
                share_obs_batch = np.stack(share_obs_batch, axis=1)
                obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys(): 
                    share_obs_batch[key] = _flatten(L, N, share_obs_batch[key])
                for key in obs_batch.keys(): 
                    obs_batch[key] = _flatten(L, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(L, N, share_obs_batch)
                obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator_ensemble(self, advantages, num_mini_batch=None, data_chunk_length=None):
        episode_length, _, num_agents = self.rewards.shape[0:3]
        n_rollout_threads = self.n_rollout_threads
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        self.shuffle_batch = True
        if self.shuffle_batch:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

            # diff mini_batch for multi-critic
            rand_critic = []
            sampler_critic = []
            for critic_id in range(self.num_critic):
                rand_critic.append(torch.randperm(batch_size).numpy())
                sampler_critic.append([rand_critic[critic_id][i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)])
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()
        else:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
            sampler_critic = []
            for critic_id in range(self.num_critic):
                sampler_critic.append(sampler_actor)
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()

        share_obs_critic = _cast(self.share_obs[:-1])
        obs_actor = _cast(self.obs[:-1])

        rnn_states_actor = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        actions_actor = _cast(self.actions)
        masks_actor = _cast(self.masks[:-1])
        active_masks_actor = _cast(self.active_masks[:-1]) 
        advantages_actor = _cast(advantages)
        action_log_probs_actor = _cast(self.action_log_probs)

        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.rnn_states_critic.shape[3:])
        masks_critic = _cast(self.masks[:-1])
        value_preds_critic = self.value_preds[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.value_preds.shape[3:])
        returns_critic = self.returns[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.returns.shape[3:])
        active_masks_critic = _cast(self.active_masks[:-1]) 

        if self.available_actions is not None:
            available_actions_actor = _cast(self.available_actions[:-1])

        # for indices in sampler:
        for indices_actor, indices_critic in zip(sampler_actor, sampler_critic):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_critic_batch = []
                obs_actor_batch = []

            rnn_states_actor_batch = []
            rnn_states_critic_batch = []
            actions_actor_batch = []
            available_actions_actor_batch = []
            value_preds_critic_batch = []
            return_critic_batch = []
            masks_actor_batch = []
            masks_critic_batch = []
            active_masks_actor_batch = []
            active_masks_critic_batch = []
            old_action_log_probs_actor_batch = []
            adv_targ_actor = []

            L, N = data_chunk_length, mini_batch_size

            for index in indices_actor:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                obs_actor_batch.append(obs_actor[ind:ind+data_chunk_length])

                actions_actor_batch.append(actions_actor[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_actor_batch.append(available_actions_actor[ind:ind+data_chunk_length])
                masks_actor_batch.append(masks_actor[ind:ind+data_chunk_length])
                active_masks_actor_batch.append(active_masks_actor[ind:ind+data_chunk_length])
                old_action_log_probs_actor_batch.append(action_log_probs_actor[ind:ind+data_chunk_length])
                adv_targ_actor.append(advantages_actor[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
            # These are all from_numpys of size (L, N, Dim)     
            obs_actor_batch = np.stack(obs_actor_batch, axis=1)
            actions_actor_batch = np.stack(actions_actor_batch, axis=1)
            if self.available_actions is not None:
                available_actions_actor_batch = np.stack(available_actions_actor_batch, axis=1)
            masks_actor_batch = np.stack(masks_actor_batch, axis=1)
            active_masks_actor_batch = np.stack(active_masks_actor_batch, axis=1)
            old_action_log_probs_actor_batch = np.stack(old_action_log_probs_actor_batch, axis=1)
            adv_targ_actor = np.stack(adv_targ_actor, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_actor_batch = _flatten(L, N, obs_actor_batch)
            actions_actor_batch = _flatten(L, N, actions_actor_batch)
            if self.available_actions is not None:
                available_actions_actor_batch = _flatten(L, N, available_actions_actor_batch)
            else:
                available_actions_batch = None
            masks_actor_batch = _flatten(L, N, masks_actor_batch)
            active_masks_actor_batch = _flatten(L, N, active_masks_actor_batch)
            old_action_log_probs_actor_batch = _flatten(L, N, old_action_log_probs_actor_batch)
            adv_targ_actor = _flatten(L, N, adv_targ_actor)

            # first stack data_chunk, then stack critic_id
            for critic_id in range(self.num_critic):
                share_obs_critic_batch_one = []
                value_preds_critic_batch_one = []
                return_critic_batch_one = []
                masks_critic_batch_one = []
                active_masks_critic_batch_one = []
                rnn_states_critic_batch_one = []
                for index in indices_critic[critic_id]:
                    ind = index * data_chunk_length
                    # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                    share_obs_critic_batch_one.append(share_obs_critic[ind:ind+data_chunk_length])
                    value_preds_critic_batch_one.append(value_preds_critic[ind:ind+data_chunk_length, critic_id])
                    return_critic_batch_one.append(returns_critic[ind:ind+data_chunk_length,critic_id])
                    masks_critic_batch_one.append(masks_critic[ind:ind+data_chunk_length])
                    active_masks_critic_batch_one.append(active_masks_critic[ind:ind+data_chunk_length])
                    rnn_states_critic_batch_one.append(rnn_states_critic[ind,critic_id])
                
                # These are all from_numpys of size (L, N, Dim)  
                share_obs_critic_batch_one = np.stack(share_obs_critic_batch_one, axis=1)

                masks_critic_batch_one = np.stack(masks_critic_batch_one, axis=1)
                active_masks_critic_batch_one = np.stack(active_masks_critic_batch_one, axis=1)
                value_preds_critic_batch_one = np.stack(value_preds_critic_batch_one, axis=1)
                return_critic_batch_one = np.stack(return_critic_batch_one, axis=1)

                # States is just a (N, -1) from_numpy
                rnn_states_critic_batch_one = np.stack(rnn_states_critic_batch_one).reshape(N, *self.rnn_states_critic.shape[4:])
                
                # Flatten the (L, N, ...) from_numpys to (L * N, ...)
                share_obs_critic_batch_one = _flatten(L, N, share_obs_critic_batch_one)
                masks_critic_batch_one = _flatten(L, N, masks_critic_batch_one)
                active_masks_critic_batch_one = _flatten(L, N, active_masks_critic_batch_one)
                value_preds_critic_batch_one = _flatten(L, N, value_preds_critic_batch_one)
                return_critic_batch_one = _flatten(L, N ,return_critic_batch_one)

                # handle num_critic
                share_obs_critic_batch.append(share_obs_critic_batch_one)
                masks_critic_batch.append(masks_critic_batch_one)
                active_masks_critic_batch.append(active_masks_critic_batch_one)
                value_preds_critic_batch.append(value_preds_critic_batch_one)
                return_critic_batch.append(return_critic_batch_one)
                rnn_states_critic_batch.append(rnn_states_critic_batch_one)
            
            # handle num_critic
            share_obs_critic_batch = np.stack(share_obs_critic_batch, axis=1)
            masks_critic_batch = np.stack(masks_critic_batch, axis=1)
            active_masks_critic_batch = np.stack(active_masks_critic_batch, axis=1)
            value_preds_critic_batch = np.stack(value_preds_critic_batch, axis=1)
            return_critic_batch = np.stack(return_critic_batch, axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch,axis=1)

            yield obs_actor_batch, rnn_states_actor_batch, actions_actor_batch, masks_actor_batch, active_masks_actor_batch, adv_targ_actor, old_action_log_probs_actor_batch, available_actions_batch, \
                  share_obs_critic_batch, rnn_states_critic_batch, masks_critic_batch, value_preds_critic_batch, return_critic_batch, active_masks_critic_batch

class SharedReplayBuffer_ensemble(object):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.num_critic = args.num_critic
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits 

        self._mixed_obs = False  # for mixed observation   

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # for mixed observation
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}
            self.share_obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
            for key in share_obs_shape:
                self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape[key].shape), dtype=np.float32)
        
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if type(share_obs_shape[-1]) == list:
                share_obs_shape = share_obs_shape[:1]

            self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # TODO: invalid num_critic
        self.rnn_states_critic = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_critic), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
               
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):

        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][self.step + 1] = share_obs[key].copy()
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.share_obs[self.step + 1] = share_obs.copy()
            self.obs[self.step + 1] = obs.copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][0] = self.share_obs[key][-1].copy()
            for key in self.obs.keys():
                self.obs[key][0] = self.obs[key][-1].copy()
        else:
            self.share_obs[0] = self.share_obs[-1].copy()
            self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1] 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                # change for ensemble
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def feed_forward_generator_ensemble(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, _, num_agents = self.rewards.shape[0:3]
        n_rollout_threads = self.n_rollout_threads
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        self.shuffle_batch = True
        if self.shuffle_batch:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

            # diff mini_batch for multi-critic
            rand_critic = []
            sampler_critic = []
            for critic_id in range(self.num_critic):
                rand_critic.append(torch.randperm(batch_size).numpy())
                sampler_critic.append([rand_critic[critic_id][i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)])
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()
        else:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
            sampler_critic = []
            for critic_id in range(self.num_critic):
                sampler_critic.append(sampler_actor)
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            obs_actor = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states_actor = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        actions_actor = self.actions[:].reshape(-1, self.actions.shape[-1])
        masks_actor = self.masks[:-1].reshape(-1, 1)
        active_masks_actor = self.active_masks[:-1].reshape(-1, 1)
        advantages_actor = advantages.reshape(-1, 1)
        action_log_probs_actor = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        
        share_obs_critic = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        masks_critic = self.masks[:-1].reshape(-1, 1)
        value_preds_critic = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[3:])
        returns_critic = self.returns[:-1].reshape(-1, *self.returns.shape[3:])
        active_masks_critic = self.active_masks[:-1].reshape(-1, 1)

        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        
        for indices_actor,indices_critic in zip(sampler_actor, sampler_critic):
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            # indices_critic : num_critic * index
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                obs_actor_batch = obs_actor[indices_actor]
            rnn_states_actor_batch = rnn_states_actor[indices_actor]
            actions_actor_batch = actions_actor[indices_actor]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices_actor]
            else:
                available_actions_batch = None
            masks_actor_batch = masks_actor[indices_actor]
            active_masks_actor_batch = active_masks_actor[indices_actor]
            old_action_log_probs_actor_batch = action_log_probs_actor[indices_actor]

            # diff critic batch
            share_obs_critic_batch = []
            rnn_states_critic_batch = []
            value_preds_critic_batch = []
            return_critic_batch = []
            masks_critic_batch = []
            active_masks_critic_batch = []
            for critic_id in range(self.num_critic):
                share_obs_critic_batch.append(share_obs_critic[indices_critic[critic_id]])
                # TODO invalid rnn_states_critic_batch
                rnn_states_critic_batch.append(rnn_states_critic[indices_critic[critic_id]])
                value_preds_critic_batch.append(value_preds_critic[indices_critic[critic_id],critic_id])
                return_critic_batch.append(returns_critic[indices_critic[critic_id],critic_id])
                masks_critic_batch.append(masks_critic[indices_critic[critic_id]])
                active_masks_critic_batch.append(active_masks_critic[indices_critic[critic_id]])
            share_obs_critic_batch = np.stack(share_obs_critic_batch,axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch,axis=1)
            value_preds_critic_batch = np.stack(value_preds_critic_batch, axis=1)
            return_critic_batch = np.stack(return_critic_batch, axis=1)
            masks_critic_batch = np.stack(masks_critic_batch, axis=1)
            active_masks_critic_batch = np.stack(active_masks_critic_batch, axis=1)

            if advantages is None:
                adv_targ_actor = None
            else:
                adv_targ_actor = advantages_actor[indices_actor]

            yield obs_actor_batch, rnn_states_actor_batch, actions_actor_batch, masks_actor_batch, active_masks_actor_batch, adv_targ_actor, old_action_log_probs_actor_batch, available_actions_batch, \
                  share_obs_critic_batch, rnn_states_critic_batch, masks_critic_batch, value_preds_critic_batch, return_critic_batch, active_masks_critic_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads*num_agents
        assert n_rollout_threads*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()
        
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key].reshape(-1, batch_size, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key].reshape(-1, batch_size, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
            obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][:-1, ind])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][:-1, ind])
                else:
                    share_obs_batch.append(share_obs[:-1, ind])
                    obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
            
            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = np.stack(share_obs_batch[key], 1)
                for key in obs_batch.keys():
                    obs_batch[key] = np.stack(obs_batch[key], 1)
            else:
                share_obs_batch = np.stack(share_obs_batch, 1)
                obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = _flatten(T, N, share_obs_batch[key])
                for key in obs_batch.keys():
                    obs_batch[key] = _flatten(T, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(T, N, share_obs_batch)
                obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                if len(self.share_obs[key].shape) == 6:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs[key].shape[3:])
                elif len(self.share_obs[key].shape) == 5:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.share_obs[key].shape[3:])
                else:
                    share_obs[key] = _cast(self.share_obs[key][:-1])
                   
            for key in self.obs.keys():
                if len(self.obs[key].shape) == 6:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs[key].shape[3:])
                elif len(self.obs[key].shape) == 5:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.obs[key].shape[3:])
                else:
                    obs[key] = _cast(self.obs[key][:-1])
        else:
            if len(self.share_obs.shape) > 4:
                share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
                obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            else:
                share_obs = _cast(self.share_obs[:-1])
                obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])       
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                else:
                    share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                    obs_batch.append(obs[ind:ind+data_chunk_length])

                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim) 
            if self._mixed_obs:
                for key in share_obs_batch.keys():  
                    share_obs_batch[key] = np.stack(share_obs_batch[key], axis=1)
                for key in obs_batch.keys():  
                    obs_batch[key] = np.stack(obs_batch[key], axis=1)
            else:        
                share_obs_batch = np.stack(share_obs_batch, axis=1)
                obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys(): 
                    share_obs_batch[key] = _flatten(L, N, share_obs_batch[key])
                for key in obs_batch.keys(): 
                    obs_batch[key] = _flatten(L, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(L, N, share_obs_batch)
                obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator_ensemble(self, advantages, num_mini_batch=None, data_chunk_length=None):
        episode_length, _, num_agents = self.rewards.shape[0:3]
        n_rollout_threads = self.n_rollout_threads
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        self.shuffle_batch = True
        if self.shuffle_batch:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

            # diff mini_batch for multi-critic
            rand_critic = []
            sampler_critic = []
            for critic_id in range(self.num_critic):
                rand_critic.append(torch.randperm(batch_size).numpy())
                sampler_critic.append([rand_critic[critic_id][i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)])
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()
        else:
            rand_actor = torch.randperm(batch_size).numpy()
            sampler_actor = [rand_actor[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
            sampler_critic = []
            for critic_id in range(self.num_critic):
                sampler_critic.append(sampler_actor)
            # sampler critic: -> num_mini_batch * num_critic * mini_batch_size
            sampler_critic = np.array(sampler_critic).transpose(1,0,2).tolist()


        share_obs_critic = _cast(self.share_obs[:-1])
        obs_actor = _cast(self.obs[:-1])

        rnn_states_actor = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        actions_actor = _cast(self.actions)
        masks_actor = _cast(self.masks[:-1])
        active_masks_actor = _cast(self.active_masks[:-1]) 
        advantages_actor = _cast(advantages)
        action_log_probs_actor = _cast(self.action_log_probs)

        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.rnn_states_critic.shape[3:])
        masks_critic = _cast(self.masks[:-1])
        value_preds_critic = self.value_preds[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.value_preds.shape[3:])
        returns_critic = self.returns[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.returns.shape[3:])
        active_masks_critic = _cast(self.active_masks[:-1]) 

        if self.available_actions is not None:
            available_actions_actor = _cast(self.available_actions[:-1])

        # for indices in sampler:
        for indices_actor, indices_critic in zip(sampler_actor, sampler_critic):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_critic_batch = []
                obs_actor_batch = []

            rnn_states_actor_batch = []
            rnn_states_critic_batch = []
            actions_actor_batch = []
            available_actions_actor_batch = []
            value_preds_critic_batch = []
            return_critic_batch = []
            masks_actor_batch = []
            masks_critic_batch = []
            active_masks_actor_batch = []
            active_masks_critic_batch = []
            old_action_log_probs_actor_batch = []
            adv_targ_actor = []

            L, N = data_chunk_length, mini_batch_size

            for index in indices_actor:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                else:
                    # share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                    obs_actor_batch.append(obs_actor[ind:ind+data_chunk_length])

                actions_actor_batch.append(actions_actor[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_actor_batch.append(available_actions_actor[ind:ind+data_chunk_length])
                masks_actor_batch.append(masks_actor[ind:ind+data_chunk_length])
                active_masks_actor_batch.append(active_masks_actor[ind:ind+data_chunk_length])
                old_action_log_probs_actor_batch.append(action_log_probs_actor[ind:ind+data_chunk_length])
                adv_targ_actor.append(advantages_actor[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
            # These are all from_numpys of size (L, N, Dim) 
            if self._mixed_obs:
                pass
            else:        
                obs_actor_batch = np.stack(obs_actor_batch, axis=1)

            actions_actor_batch = np.stack(actions_actor_batch, axis=1)
            if self.available_actions is not None:
                available_actions_actor_batch = np.stack(available_actions_actor_batch, axis=1)
            masks_actor_batch = np.stack(masks_actor_batch, axis=1)
            active_masks_actor_batch = np.stack(active_masks_actor_batch, axis=1)
            old_action_log_probs_actor_batch = np.stack(old_action_log_probs_actor_batch, axis=1)
            adv_targ_actor = np.stack(adv_targ_actor, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                pass
            else:
                obs_actor_batch = _flatten(L, N, obs_actor_batch)
            actions_actor_batch = _flatten(L, N, actions_actor_batch)
            if self.available_actions is not None:
                available_actions_actor_batch = _flatten(L, N, available_actions_actor_batch)
            else:
                available_actions_batch = None
            masks_actor_batch = _flatten(L, N, masks_actor_batch)
            active_masks_actor_batch = _flatten(L, N, active_masks_actor_batch)
            old_action_log_probs_actor_batch = _flatten(L, N, old_action_log_probs_actor_batch)
            adv_targ_actor = _flatten(L, N, adv_targ_actor)

            # first stack data_chunk, then stack critic_id
            for critic_id in range(self.num_critic):
                share_obs_critic_batch_one = []
                value_preds_critic_batch_one = []
                return_critic_batch_one = []
                masks_critic_batch_one = []
                active_masks_critic_batch_one = []
                rnn_states_critic_batch_one = []
                for index in indices_critic[critic_id]:
                    ind = index * data_chunk_length
                    # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                    if self._mixed_obs:
                        for key in share_obs.keys():
                            share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                        for key in obs.keys():
                            obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                    else:
                        share_obs_critic_batch_one.append(share_obs_critic[ind:ind+data_chunk_length])

                    value_preds_critic_batch_one.append(value_preds_critic[ind:ind+data_chunk_length, critic_id])
                    return_critic_batch_one.append(returns_critic[ind:ind+data_chunk_length,critic_id])
                    masks_critic_batch_one.append(masks_critic[ind:ind+data_chunk_length])
                    active_masks_critic_batch_one.append(active_masks_critic[ind:ind+data_chunk_length])
                    rnn_states_critic_batch_one.append(rnn_states_critic[ind,critic_id])
                
                # These are all from_numpys of size (L, N, Dim) 
                if self._mixed_obs:
                    pass
                else:        
                    share_obs_critic_batch_one = np.stack(share_obs_critic_batch_one, axis=1)

                masks_critic_batch_one = np.stack(masks_critic_batch_one, axis=1)
                active_masks_critic_batch_one = np.stack(active_masks_critic_batch_one, axis=1)
                value_preds_critic_batch_one = np.stack(value_preds_critic_batch_one, axis=1)
                return_critic_batch_one = np.stack(return_critic_batch_one, axis=1)

                # States is just a (N, -1) from_numpy
                rnn_states_critic_batch_one = np.stack(rnn_states_critic_batch_one).reshape(N, *self.rnn_states_critic.shape[4:])
                
                # Flatten the (L, N, ...) from_numpys to (L * N, ...)
                if self._mixed_obs:
                    pass
                else:
                    share_obs_critic_batch_one = _flatten(L, N, share_obs_critic_batch_one)
                masks_critic_batch_one = _flatten(L, N, masks_critic_batch_one)
                active_masks_critic_batch_one = _flatten(L, N, active_masks_critic_batch_one)
                value_preds_critic_batch_one = _flatten(L, N, value_preds_critic_batch_one)
                return_critic_batch_one = _flatten(L, N ,return_critic_batch_one)

                # handle num_critic
                share_obs_critic_batch.append(share_obs_critic_batch_one)
                masks_critic_batch.append(masks_critic_batch_one)
                active_masks_critic_batch.append(active_masks_critic_batch_one)
                value_preds_critic_batch.append(value_preds_critic_batch_one)
                return_critic_batch.append(return_critic_batch_one)
                rnn_states_critic_batch.append(rnn_states_critic_batch_one)
            
            # handle num_critic
            share_obs_critic_batch = np.stack(share_obs_critic_batch, axis=1)
            masks_critic_batch = np.stack(masks_critic_batch, axis=1)
            active_masks_critic_batch = np.stack(active_masks_critic_batch, axis=1)
            value_preds_critic_batch = np.stack(value_preds_critic_batch, axis=1)
            return_critic_batch = np.stack(return_critic_batch, axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch,axis=1)

            yield obs_actor_batch, rnn_states_actor_batch, actions_actor_batch, masks_actor_batch, active_masks_actor_batch, adv_targ_actor, old_action_log_probs_actor_batch, available_actions_batch, \
                  share_obs_critic_batch, rnn_states_critic_batch, masks_critic_batch, value_preds_critic_batch, return_critic_batch, active_masks_critic_batch

class SharedReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits 

        self._mixed_obs = False  # for mixed observation   

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # for mixed observation
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}
            self.share_obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
            for key in share_obs_shape:
                self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape[key].shape), dtype=np.float32)
        
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if type(share_obs_shape[-1]) == list:
                share_obs_shape = share_obs_shape[:1]

            self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
       
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
               
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):

        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][self.step + 1] = share_obs[key].copy()
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.share_obs[self.step + 1] = share_obs.copy()
            self.obs[self.step + 1] = obs.copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][0] = self.share_obs[key][-1].copy()
            for key in self.obs.keys():
                self.obs[key][0] = self.obs[key][-1].copy()
        else:
            self.share_obs[0] = self.share_obs[-1].copy()
            self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1] 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads*num_agents
        assert n_rollout_threads*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()
        
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key].reshape(-1, batch_size, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key].reshape(-1, batch_size, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
            obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][:-1, ind])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][:-1, ind])
                else:
                    share_obs_batch.append(share_obs[:-1, ind])
                    obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
            
            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = np.stack(share_obs_batch[key], 1)
                for key in obs_batch.keys():
                    obs_batch[key] = np.stack(obs_batch[key], 1)
            else:
                share_obs_batch = np.stack(share_obs_batch, 1)
                obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = _flatten(T, N, share_obs_batch[key])
                for key in obs_batch.keys():
                    obs_batch[key] = _flatten(T, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(T, N, share_obs_batch)
                obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                if len(self.share_obs[key].shape) == 6:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs[key].shape[3:])
                elif len(self.share_obs[key].shape) == 5:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.share_obs[key].shape[3:])
                else:
                    share_obs[key] = _cast(self.share_obs[key][:-1])
                   
            for key in self.obs.keys():
                if len(self.obs[key].shape) == 6:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs[key].shape[3:])
                elif len(self.obs[key].shape) == 5:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.obs[key].shape[3:])
                else:
                    obs[key] = _cast(self.obs[key][:-1])
        else:
            if len(self.share_obs.shape) > 4:
                share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
                obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            else:
                share_obs = _cast(self.share_obs[:-1])
                obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])       
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                else:
                    share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                    obs_batch.append(obs[ind:ind+data_chunk_length])

                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim) 
            if self._mixed_obs:
                for key in share_obs_batch.keys():  
                    share_obs_batch[key] = np.stack(share_obs_batch[key], axis=1)
                for key in obs_batch.keys():  
                    obs_batch[key] = np.stack(obs_batch[key], axis=1)
            else:        
                share_obs_batch = np.stack(share_obs_batch, axis=1)
                obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys(): 
                    share_obs_batch[key] = _flatten(L, N, share_obs_batch[key])
                for key in obs_batch.keys(): 
                    obs_batch[key] = _flatten(L, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(L, N, share_obs_batch)
                obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
