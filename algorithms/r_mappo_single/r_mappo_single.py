import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt


class R_MAPPO():
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.policy = policy

        self._recurrent_policy = args.recurrent_policy
        self._naive_recurrent_policy = args.naive_recurrent_policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_huber_loss = args.use_huber_loss
        self.huber_delta = args.huber_delta

        self.use_popart = args.use_popart
        self.use_value_active_masks = args.use_value_active_masks

        if self.use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def ppo_update(self, sample):
        share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
        adv_targ = adv_targ.to(self.device)
        active_masks_batch = active_masks_batch.to(self.device)
        value_preds_batch = value_preds_batch.to(self.device)
        return_batch = return_batch.to(self.device)
        active_masks_batch = active_masks_batch.to(self.device)

        # policy loss
        # Reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs_batch, recurrent_hidden_states_batch, actions_batch, masks_batch, active_masks_batch)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        kl_divergence = torch.exp(
            old_action_log_probs_batch) * (old_action_log_probs_batch - action_log_probs)
        kl_loss = (kl_divergence * active_masks_batch).sum() / \
            active_masks_batch.sum()

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = (-torch.min(surr1, surr2) *
                       active_masks_batch).sum() / active_masks_batch.sum()

        # value loss
        values = self.policy.get_value(
            share_obs_batch, recurrent_hidden_states_critic_batch, masks_batch)

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

        # update common and action network
        self.policy.optimizer.zero_grad()

        (action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.model.parameters())

        self.policy.optimizer.step()

        return value_loss, action_loss, dist_entropy, grad_norm, kl_loss, ratio

    def separated_update(self, agent_id, buffer):
        if self.use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                torch.tensor(buffer.value_preds[:-1])).cpu().numpy()
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        kl_loss_epoch = 0
        ratio_epoch = 0

        # policy phase
        for _ in range(self.ppo_epoch):

            if self._recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._naive_recurrent_policy:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, action_loss, dist_entropy, grad_norm, kl_loss, ratio = self.ppo_update(
                    sample)

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                kl_loss_epoch += kl_loss.item()
                ratio_epoch += ratio.mean()
                value_loss_epoch += value_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        kl_loss_epoch /= num_updates
        ratio_epoch /= num_updates
        value_loss_epoch /= num_updates

        return value_loss_epoch, grad_norm_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, kl_loss_epoch, ratio_epoch

    def single_update(self, agent_id, buffer):
        if self.use_popart:
            advantages = buffer.returns[:-1, :, agent_id] - self.value_normalizer.denormalize(
                torch.tensor(buffer.value_preds[:-1, :, agent_id])).cpu().numpy()
        else:
            advantages = buffer.returns[:-1, :, agent_id] - \
                buffer.value_preds[:-1, :, agent_id]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        kl_loss_epoch = 0
        ratio_epoch = 0

        # policy phase
        for _ in range(self.ppo_epoch):
            if self._recurrent_policy:
                data_generator = buffer.single_recurrent_generator(
                    agent_id, advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._naive_recurrent_policy:
                data_generator = buffer.single_naive_recurrent_generator(
                    agent_id, advantages, self.num_mini_batch)
            else:
                data_generator = buffer.single_feed_forward_generator(
                    agent_id, advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, action_loss, dist_entropy, grad_norm, kl_loss, ratio = self.ppo_update(
                    sample)

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                kl_loss_epoch += kl_loss.item()
                ratio_epoch += ratio.mean()
                value_loss_epoch += value_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        kl_loss_epoch /= num_updates
        ratio_epoch /= num_updates
        value_loss_epoch /= num_updates

        return value_loss_epoch, grad_norm_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, kl_loss_epoch, ratio_epoch

    def shared_update(self, buffer):
        if self.use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                torch.tensor(buffer.value_preds[:-1])).cpu().numpy()
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        kl_loss_epoch = 0
        ratio_epoch = 0

        # policy phase
        for _ in range(self.ppo_epoch):

            if self._recurrent_policy:
                data_generator = buffer.shared_recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._naive_recurrent_policy:
                data_generator = buffer.shared_naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = buffer.shared_feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, action_loss, dist_entropy, grad_norm, kl_loss, ratio = self.ppo_update(
                    sample)

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                grad_norm_epoch += grad_norm
                kl_loss_epoch += kl_loss.item()
                ratio_epoch += ratio.mean()
                value_loss_epoch += value_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        kl_loss_epoch /= num_updates
        ratio_epoch /= num_updates
        value_loss_epoch /= num_updates

        return value_loss_epoch, grad_norm_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch, kl_loss_epoch, ratio_epoch

    def prep_training(self):
        self.policy.model.train()

    def prep_rollout(self):
        self.policy.model.eval()
