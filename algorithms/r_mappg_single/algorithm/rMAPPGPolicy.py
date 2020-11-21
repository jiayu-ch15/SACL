import numpy as np
import torch
from algorithms.r_mappg_single.algorithm.r_actor_critic import R_Model
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
        self.act_space = action_space

        self.model = R_Model(args, self.obs_space, self.share_obs_space,
                            self.act_space, self.device, cat_self).to(self.device)


        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def act(self, share_obs, obs, actor_hidden_states, critic_hidden_states, masks, available_actions=None, deterministic=False):
        action_out, action_log_probs_out, actor_hidden_states = self.model.get_actions(
            obs, actor_hidden_states, masks, available_actions, deterministic)
        value, critic_hidden_states = self.model.get_value(
            share_obs, critic_hidden_states, masks)
        return value, action_out, action_log_probs_out, actor_hidden_states, critic_hidden_states

    def get_value(self, share_obs, critic_hidden_states, masks):
        value, _ = self.model.get_value(share_obs, critic_hidden_states, masks)
        return value

    def get_value_and_logprobs(self, share_obs, obs, actor_hidden_states, critic_hidden_states, masks, available_actions=None, deterministic=False):
        action_out, action_log_probs_out, actor_hidden_states = self.model.get_actions(
            obs, actor_hidden_states, masks, available_actions, deterministic)
        value, critic_hidden_states = self.model.get_value(
            share_obs, critic_hidden_states, masks)
        return value, action_log_probs_out

    def get_actions(self, obs, actor_hidden_states, masks, available_actions=None, deterministic=False):
        action_out, action_log_probs_out, actor_hidden_states = self.model.get_actions(
            obs, actor_hidden_states, masks, available_actions, deterministic)
        return action_out, action_log_probs_out, actor_hidden_states

    def evaluate_actions(self, obs, actor_hidden_states, action, masks, active_masks=None):
        action_log_probs_out, dist_entropy_out, _ = self.model.evaluate_actions(
            obs, actor_hidden_states, action, masks, active_masks)

        return action_log_probs_out, dist_entropy_out
