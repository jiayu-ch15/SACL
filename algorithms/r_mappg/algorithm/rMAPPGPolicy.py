import numpy as np
import torch
from algorithms.r_mappg.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class R_MAPPGPolicy:
    def __init__(self, args, obs_space, share_obs_space, action_space, cat_self=True, device=torch.device("cpu"), train=True):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = action_space

        self.actor = R_Actor(args, self.obs_space,
                            self.act_space, self.device).to(self.device)
        self.critic = R_Critic(args, self.share_obs_space, cat_self,
                            self.device).to(self.device)

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(
            ), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(
            ), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer,
                               episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer,
                               episode, episodes, self.critic_lr)

    def act(self, share_obs, obs, actor_hidden_states, critic_hidden_states, masks, available_actions=None, deterministic=False):
        action_out, action_log_probs_out, actor_hidden_states = self.actor(
            obs, actor_hidden_states, masks, available_actions, deterministic)
        value, critic_hidden_states = self.critic(
            share_obs, critic_hidden_states, masks)
        return value, action_out, action_log_probs_out, actor_hidden_states, critic_hidden_states

    def get_value(self, share_obs, critic_hidden_states, masks):
        value, _ = self.critic(share_obs, critic_hidden_states, masks)
        return value

    def get_policy_value_and_logprobs(self, obs, actor_hidden_states, masks, available_actions=None, deterministic=False):
        value, action_log_probs = self.actor.get_value_and_logprobs(obs, actor_hidden_states, masks, available_actions, deterministic)
        return value, action_log_probs

    def get_actions(self, obs, actor_hidden_states, masks, available_actions=None, deterministic=False):
        action_out, action_log_probs_out, actor_hidden_states = self.actor(
            obs, actor_hidden_states, masks, available_actions, deterministic)
        return action_out, action_log_probs_out, actor_hidden_states

    def evaluate_actions(self, obs, actor_hidden_states, action, masks, active_masks=None):
        action_log_probs_out, dist_entropy_out, _ = self.actor.evaluate_actions(
            obs, actor_hidden_states, action, masks, active_masks)

        return action_log_probs_out, dist_entropy_out
