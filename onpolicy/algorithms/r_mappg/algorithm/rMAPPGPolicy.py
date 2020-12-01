import numpy as np
import torch
from algorithms.r_mappg.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class R_MAPPGPolicy:
    def __init__(self, args, obs_space, share_obs_space, action_space, device=torch.device("cpu"), cat_self=True):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device, cat_self)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_policy_values_and_logprobs(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        values, action_log_probs = self.actor.get_values_and_logprobs(obs, rnn_states_actor, masks, available_actions, deterministic)
        return values, action_log_probs

    def get_values(self, share_obs, rnn_states_critic, masks):
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        action_log_probs, dist_entropy, _ = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

