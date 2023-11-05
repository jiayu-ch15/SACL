import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule

class R_MAPPOPolicy_ensemble:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_critic = args.num_critic

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = [R_Critic(args, self.share_obs_space, self.device) for _ in range(self.num_critic)]
        # self.critic = [R_Critic(args, self.share_obs_space, self.device)] * 3
        self.critic = nn.ModuleList(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = [torch.optim.Adam(self.critic[critic_id].parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay) for critic_id in range(self.num_critic)]

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        # values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        values_all = []
        # TODO invalid rnn_states_critic
        for critic_one in self.critic:
            values_one, rnn_states_critic = critic_one(share_obs, rnn_states_critic, masks)
            values_all.append(values_one)
        values_all = torch.concat(values_all,dim=1)
        return values_all, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        values_all = []
        for critic_one in self.critic:
            values_one, _ = critic_one(share_obs, rnn_states_critic, masks)
            values_all.append(values_one)
        values_all = torch.concat(values_all,dim=1)
        return values_all

    # for ensemble
    def get_values_seperate(self, share_obs, rnn_states_critic, masks, critic_id):
        values, _ = self.critic[critic_id](share_obs, rnn_states_critic, masks)
        return values[:,0]

    # for ensemble, update actor and critic seperately
    def only_evaluate_actions(self, obs, rnn_states_actor, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        return action_log_probs, dist_entropy, policy_values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values_all = []
        for critic_one in self.critic:
            values_one, _ = critic_one(share_obs, rnn_states_critic, masks)
            values_all.append(values_one)
        values_all = torch.concat(values_all,dim=1)
        return values_all, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

class R_MAPPOPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
