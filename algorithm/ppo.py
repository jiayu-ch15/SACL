import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.util import get_gard_norm
from utils.popart import PopArt


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 data_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 weight_decay=None,
                 max_grad_norm=None,
                 use_max_grad_norm=True,
                 use_clipped_value_loss=True,
                 use_common_layer=False,
                 use_huber_loss=False,
                 huber_delta=2,
                 use_popart=True,
                 use_value_active_masks=False,
                 device=torch.device("cpu")):

        self.step = 0
        self.device = device
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.data_chunk_length = data_chunk_length

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_max_grad_norm = use_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_common_layer = use_common_layer
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.use_popart = use_popart
        self.use_value_active_masks = use_value_active_masks
        if self.use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def ppo_update(self, sample, turn_on=True):

        share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ = sample

        old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
        adv_targ = adv_targ.to(self.device)
        value_preds_batch = value_preds_batch.to(self.device)
        return_batch = return_batch.to(self.device)
        active_masks_batch = active_masks_batch.to(self.device)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(share_obs_batch,
                                                                                          obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, masks_batch, active_masks_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        KL_divloss = nn.KLDivLoss(reduction='batchmean')(
            old_action_log_probs_batch, torch.exp(action_log_probs))

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = (-torch.min(surr1, surr2) *
                       active_masks_batch).sum() / active_masks_batch.sum()

        if self.use_popart:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            error_clipped = self.value_normalizer(
                return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / \
                active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        self.optimizer.zero_grad()

        if self.use_common_layer:
            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()
        else:
            (value_loss * self.value_loss_coef).backward()
            if turn_on == True:
                (action_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.actor_critic.parameters())

        self.optimizer.step()

        return value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio

    def separated_update(self, agent_id, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1] - self.value_normalizer.denormalize(
                torch.tensor(rollouts.value_preds[:-1])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        KL_divloss_epoch = 0
        ratio_epoch = 0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = self.ppo_update(
                    sample, turn_on)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                KL_divloss_epoch += KL_divloss.item()
                ratio_epoch += ratio.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        KL_divloss_epoch /= num_updates
        ratio_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, KL_divloss_epoch, ratio_epoch

    def single_update(self, agent_id, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1, :, agent_id] - self.value_normalizer.denormalize(
                torch.tensor(rollouts.value_preds[:-1, :, agent_id])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1, :, agent_id] - \
                rollouts.value_preds[:-1, :, agent_id]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        KL_divloss_epoch = 0
        ratio_epoch = 0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.single_recurrent_generator(
                    agent_id, advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.single_naive_recurrent_generator(
                    agent_id, advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.single_feed_forward_generator(
                    agent_id, advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = self.ppo_update(
                    sample, turn_on)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                KL_divloss_epoch += KL_divloss.item()
                ratio_epoch += ratio.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        KL_divloss_epoch /= num_updates
        ratio_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, KL_divloss_epoch, ratio_epoch

    def shared_update(self, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1] - self.value_normalizer.denormalize(
                torch.tensor(rollouts.value_preds[:-1])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        KL_divloss_epoch = 0
        ratio_epoch = 0

        for _ in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.shared_recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.shared_naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.shared_feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, action_loss, dist_entropy, grad_norm, KL_divloss, ratio = self.ppo_update(
                    sample, turn_on)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                KL_divloss_epoch += KL_divloss.item()
                ratio_epoch += ratio.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        KL_divloss_epoch /= num_updates
        ratio_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, KL_divloss_epoch, ratio_epoch
