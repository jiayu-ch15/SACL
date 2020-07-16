import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def huber_loss(e, d):
    a = (abs(e)<=d).float()
    b = (e>d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.train = True

        self.running_mean = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).cuda()
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).cuda()
        self.debiasing_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False).cuda()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector.to(torch.float)

        if self.train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()            
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        return input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

class PPO():
    def __init__(self,                 
                 actor_critic,
                 agent_id,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 data_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 logger = None,
                 lr=None,
                 eps=None,
                 weight_decay=None,
                 max_grad_norm=None,
                 use_max_grad_norm=True,
                 use_clipped_value_loss=True,
                 common_layer=False,
                 use_huber_loss = False,
                 huber_delta=2,
                 use_popart = True):

        self.agent_id = agent_id
        self.step = 0
        self.logger = logger
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
        self.common_layer = common_layer
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.use_popart = use_popart
        self.value_normalizer = PopArt(1)

    def update(self, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1] - self.value_normalizer.denormalize(rollouts.value_preds[:-1])
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent or self.actor_critic.is_lstm:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, recurrent_c_states_batch, recurrent_c_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _, _, _ = self.actor_critic.evaluate_actions(self.agent_id, share_obs_batch, 
                obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, recurrent_c_states_batch, recurrent_c_states_critic_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))
                
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        error_clipped = return_batch - value_pred_clipped
                        value_losses_clipped = huber_loss(error_clipped,self.huber_delta)
                        error = return_batch - values
                        value_losses = huber_loss(error,self.huber_delta)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch) - values
                        else:
                            error = return_batch - values
                        value_loss = huber_loss(error,self.huber_delta).mean()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (self.value_normalizer(return_batch) - values).pow(2).mean()
                        else:
                            value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                self.optimizer.zero_grad()
                
                if self.common_layer:
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                else:              
                    (value_loss * self.value_loss_coef).backward()
                    if turn_on == True:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                
                grad_norm = 0
                for name,param in self.actor_critic.named_parameters():
                    grad_norm += param.grad.norm()
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                if self.logger is not None:
                    rew = []
                    for i in range(rollouts.rewards.size()[1]):
                        rew.append(rollouts.rewards[:,i,:].sum().cpu().numpy())
                    self.logger.add_scalars('agent%i/mean_episode_reward' % self.agent_id,
                        {'mean_episode_reward': np.mean(np.array(rew)/(rollouts.rewards.size()[0]))},
                        self.step)

                    self.logger.add_scalars('agent%i/value_loss' % self.agent_id,
                        {'value_loss': value_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/action_loss' % self.agent_id,
                        {'action_loss': action_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/dist_entropy' % self.agent_id,
                        {'dist_entropy': dist_entropy},
                        self.step)
                    self.logger.add_scalars('agent%i/KL_divloss' % self.agent_id,
                        {'KL_divloss': KL_divloss},
                        self.step)
                    self.logger.add_scalars('agent%i/grad_norm' % self.agent_id,
                        {'grad_norm': grad_norm},
                        self.step)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                self.step += 1
                
                
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
