import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_agents, episode_length, n_rollout_threads, obs_shape, action_space,
                 recurrent_hidden_state_size):
        
        if len(obs_shape) == 3:
            self.share_obs = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, obs_shape[0] * num_agents, obs_shape[1], obs_shape[2])
            self.obs = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, *obs_shape)
        else:
            self.share_obs = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, obs_shape[0] * num_agents)
            self.obs = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, obs_shape[0])
               
        self.recurrent_hidden_states = torch.zeros(
            episode_length + 1, n_rollout_threads, num_agents, recurrent_hidden_state_size)
        self.recurrent_hidden_states_critic = torch.zeros(
            episode_length + 1, n_rollout_threads, num_agents, recurrent_hidden_state_size)
            
        self.rewards = torch.zeros(episode_length, n_rollout_threads, num_agents, 1)
        self.value_preds = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, 1)
        self.returns = torch.zeros(episode_length + 1, n_rollout_threads, num_agents, 1)
        self.action_log_probs = torch.zeros(episode_length, n_rollout_threads, num_agents, 1)
        
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        elif action_space.__class__.__name__ == "Box":
            action_shape = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            action_shape = action_space.shape[0]
        else:#agar
            action_shape = action_space[0].shape[0] + 1
        self.actions = torch.zeros(episode_length, n_rollout_threads, num_agents, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(episode_length + 1, n_rollout_threads, num_agents, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(episode_length + 1, n_rollout_threads, num_agents, 1)
        
        self.high_masks = torch.ones(episode_length + 1, n_rollout_threads, num_agents, 1)

        self.episode_length = episode_length
        self.step = 0

    def to(self, device):
        self.share_obs = self.share_obs.to(device)
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_hidden_states_critic = self.recurrent_hidden_states_critic.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.high_masks = self.high_masks.to(device)

    def insert(self, share_obs, obs, recurrent_hidden_states, recurrent_hidden_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, high_masks=None):
        self.share_obs[self.step + 1].copy_(share_obs)
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.recurrent_hidden_states_critic[self.step + 1].copy_(recurrent_hidden_states_critic)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if high_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        if high_masks is not None:
            self.high_masks[self.step + 1].copy_(high_masks)

        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self,
                        agent_id,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        use_popart=True,
                        value_normalizer=None):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1,:,agent_id] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    if use_popart:
                        delta = self.rewards[step,:,agent_id] + gamma * value_normalizer.denormalize(self.value_preds[
                        step + 1,:,agent_id]) * self.masks[step + 1,:,agent_id] - value_normalizer.denormalize(self.value_preds[step,:,agent_id])
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:,agent_id] * gae
                        gae = gae * self.bad_masks[step + 1,:,agent_id]
                        self.returns[step,:,agent_id] = gae + value_normalizer.denormalize(self.value_preds[step,:,agent_id])
                    else:
                        delta = self.rewards[step,:,agent_id] + gamma * self.value_preds[
                            step + 1,:,agent_id] * self.masks[step + 1,:,agent_id] - self.value_preds[step,:,agent_id]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:,agent_id] * gae
                        gae = gae * self.bad_masks[step + 1,:,agent_id]
                        self.returns[step,:,agent_id] = gae + self.value_preds[step,:,agent_id]
            else:
                self.returns[-1,:,agent_id] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    if use_popart:
                        self.returns[step,:,agent_id] = (self.returns[step + 1,:,agent_id] * \
                        gamma * self.masks[step + 1,:,agent_id] + self.rewards[step,:,agent_id]) * self.bad_masks[step + 1,:,agent_id] \
                        + (1 - self.bad_masks[step + 1,:,agent_id]) * value_normalizer.denormalize(self.value_preds[step,:,agent_id])
                    else:
                        self.returns[step,:,agent_id] = (self.returns[step + 1,:,agent_id] * \
                            gamma * self.masks[step + 1,:,agent_id] + self.rewards[step,:,agent_id]) * self.bad_masks[step + 1,:,agent_id] \
                            + (1 - self.bad_masks[step + 1,:,agent_id]) * self.value_preds[step,:,agent_id]
        else:
            if use_gae:
                self.value_preds[-1,:,agent_id] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    if use_popart:
                        delta = self.rewards[step,:,agent_id] + gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1,:,agent_id]) * self.masks[step + 1,:,agent_id] - value_normalizer.denormalize(self.value_preds[step,:,agent_id])
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:,agent_id] * gae                       
                        self.returns[step,:,agent_id] = gae + value_normalizer.denormalize(self.value_preds[step,:,agent_id])
                    else:
                        delta = self.rewards[step,:,agent_id] + gamma * self.value_preds[step + 1,:,agent_id] * self.masks[step + 1,:,agent_id] - self.value_preds[step,:,agent_id]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:,agent_id] * gae
                        self.returns[step,:,agent_id] = gae + self.value_preds[step,:,agent_id]
            else:
                self.returns[-1,:,agent_id] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step,:,agent_id] = self.returns[step + 1,:,agent_id] * \
                            gamma * self.masks[step + 1,:,agent_id] + self.rewards[step,:,agent_id]

    def feed_forward_generator(self, agent_id, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = self.share_obs[:-1,:,agent_id].view(-1, *self.share_obs.size()[3:])[indices]
            obs_batch = self.obs[:-1,:,agent_id].view(-1, *self.obs.size()[3:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1,:,agent_id].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            recurrent_hidden_states_critic_batch = self.recurrent_hidden_states_critic[:-1,:,agent_id].view(
                -1, self.recurrent_hidden_states_critic.size(-1))[indices]
            actions_batch = self.actions[:,:,agent_id].view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1,:,agent_id].view(-1, 1)[indices]
            return_batch = self.returns[:-1,:,agent_id].view(-1, 1)[indices]
            masks_batch = self.masks[:-1,:,agent_id].view(-1, 1)[indices]
            high_masks_batch = self.high_masks[:-1,:,agent_id].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs[:,:,agent_id].view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks, old_action_log_probs_batch, adv_targ
            
    def feed_forward_generator_share(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.size()[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length* num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]           
            share_obs_batch = self.share_obs[:-1].view(-1, *self.share_obs.size()[3:])[indices]            
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[3:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            recurrent_hidden_states_critic_batch = self.recurrent_hidden_states_critic[:-1].view(
                -1, self.recurrent_hidden_states_critic.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            high_masks_batch = self.high_masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks, old_action_log_probs_batch, adv_targ

    def naive_recurrent_generator(self, agent_id, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.size(1)
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads)
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind, agent_id])
                obs_batch.append(self.obs[:-1, ind, agent_id])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind, agent_id])
                recurrent_hidden_states_critic_batch.append(
                    self.recurrent_hidden_states_critic[0:1, ind, agent_id])
                actions_batch.append(self.actions[:, ind, agent_id])
                value_preds_batch.append(self.value_preds[:-1, ind, agent_id])
                return_batch.append(self.returns[:-1, ind, agent_id])
                masks_batch.append(self.masks[:-1, ind, agent_id])
                high_masks_batch.append(self.high_masks[:-1, ind, agent_id])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind, agent_id])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            high_masks_batch = torch.stack(high_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            share_obs_batch = _flatten_helper(T, N, share_obs_batch)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            high_masks_batch = _flatten_helper(T, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
            
    def naive_recurrent_generator_share(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.size()[0:3]
        batch_size = n_rollout_threads*num_agents
        assert n_rollout_threads*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size)
        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs.view(-1, batch_size, *self.share_obs.size()[3:])[:-1, ind])
                obs_batch.append(self.obs.view(-1, batch_size, *self.obs.size()[3:])[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states.view(-1, batch_size, self.recurrent_hidden_states.size(-1))[0:1, ind])
                recurrent_hidden_states_critic_batch.append(
                    self.recurrent_hidden_states_critic.view(-1, batch_size, self.recurrent_hidden_states_critic.size(-1))[0:1, ind])
                actions_batch.append(self.actions.view(-1, batch_size, self.actions.size(-1))[:, ind])
                value_preds_batch.append(self.value_preds.view(-1, batch_size, 1)[:-1, ind])
                return_batch.append(self.returns.view(-1, batch_size, 1)[:-1, ind])
                masks_batch.append(self.masks.view(-1, batch_size, 1)[:-1, ind])
                high_masks_batch.append(self.high_masks.view(-1, batch_size, 1)[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs.view(-1, batch_size, 1)[:, ind])
                adv_targ.append(advantages.view(-1, batch_size, 1)[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            high_masks_batch = torch.stack(high_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            share_obs_batch = _flatten_helper(T, N, share_obs_batch)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            high_masks_batch = _flatten_helper(T, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
                
                
    def recurrent_generator(self, agent_id, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.size()[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length #[C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(data_chunks)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]                
                share_obs_batch.append(torch.transpose(self.share_obs[:-1,:,agent_id],0,1).reshape(-1, *self.share_obs.size()[3:])[ind:ind+data_chunk_length])
                obs_batch.append(torch.transpose(self.obs[:-1,:,agent_id],0,1).reshape(-1, *self.obs.size()[3:])[ind:ind+data_chunk_length])
                actions_batch.append(torch.transpose(self.actions[:,:,agent_id],0,1).reshape(-1, self.actions.size(-1))[ind:ind+data_chunk_length])
                value_preds_batch.append(torch.transpose(self.value_preds[:-1,:,agent_id],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                return_batch.append(torch.transpose(self.returns[:-1,:,agent_id],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                masks_batch.append(torch.transpose(self.masks[:-1,:,agent_id],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                high_masks_batch.append(torch.transpose(self.high_masks[:-1,:,agent_id],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(torch.transpose(self.action_log_probs[:,:,agent_id],0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                adv_targ.append(torch.transpose(advantages,0,1).reshape(-1, 1)[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                recurrent_hidden_states_batch.append(torch.transpose(self.recurrent_hidden_states[:-1,:,agent_id],0,1).reshape(
                      -1, self.recurrent_hidden_states.size(-1))[ind])
                recurrent_hidden_states_critic_batch.append(torch.transpose(self.recurrent_hidden_states_critic[:-1,:,agent_id],0,1).reshape(
                      -1, self.recurrent_hidden_states_critic.size(-1))[ind])
                      
            L, N =  data_chunk_length, mini_batch_size
                        
            # These are all tensors of size (L, N, Dim)
            share_obs_batch = torch.stack(share_obs_batch)         
            obs_batch = torch.stack(obs_batch)
            
            actions_batch = torch.stack(actions_batch)
            value_preds_batch = torch.stack(value_preds_batch)
            return_batch = torch.stack(return_batch)
            masks_batch = torch.stack(masks_batch)
            high_masks_batch = torch.stack(high_masks_batch)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch)
            adv_targ = torch.stack(adv_targ)

            # States is just a (N, -1) tensor
            
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch).view(N, -1)

            # Flatten the (L, N, ...) tensors to (L * N, ...)
            share_obs_batch = _flatten_helper(L, N, share_obs_batch)
            obs_batch = _flatten_helper(L, N, obs_batch)
            actions_batch = _flatten_helper(L, N, actions_batch)
            value_preds_batch = _flatten_helper(L, N, value_preds_batch)
            return_batch = _flatten_helper(L, N, return_batch)
            masks_batch = _flatten_helper(L, N, masks_batch)
            high_masks_batch = _flatten_helper(L, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(L, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(L, N, adv_targ)
            
            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
 
            
    def recurrent_generator_share(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.size()[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length #[C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(data_chunks)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N T M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]                
                share_obs_batch.append(self.share_obs[:-1].transpose(0,1).transpose(1,2).reshape(-1, *self.share_obs.size()[3:])[ind:ind+data_chunk_length])
                obs_batch.append(self.obs[:-1].transpose(0,1).transpose(1,2).reshape(-1, *self.obs.size()[3:])[ind:ind+data_chunk_length])
                actions_batch.append(self.actions.transpose(0,1).transpose(1,2).reshape(-1, self.actions.size(-1))[ind:ind+data_chunk_length])
                value_preds_batch.append(self.value_preds[:-1].transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                return_batch.append(self.returns[:-1].transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                masks_batch.append(self.masks[:-1].transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                high_masks_batch.append(self.high_masks[:-1].transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(self.action_log_probs.transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                adv_targ.append(advantages.transpose(0,1).transpose(1,2).reshape(-1, 1)[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[:-1].transpose(0,1).transpose(1,2).reshape(
                      -1, self.recurrent_hidden_states.size(-1))[ind])
                recurrent_hidden_states_critic_batch.append(self.recurrent_hidden_states_critic[:-1].transpose(0,1).transpose(1,2).reshape(
                      -1, self.recurrent_hidden_states_critic.size(-1))[ind])
                      
            L, N =  data_chunk_length, mini_batch_size
                        
            # These are all tensors of size (L, N, Dim)
            share_obs_batch = torch.stack(share_obs_batch)         
            obs_batch = torch.stack(obs_batch)
            
            actions_batch = torch.stack(actions_batch)
            value_preds_batch = torch.stack(value_preds_batch)
            return_batch = torch.stack(return_batch)
            masks_batch = torch.stack(masks_batch)
            high_masks_batch = torch.stack(high_masks_batch)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch)
            adv_targ = torch.stack(adv_targ)

            # States is just a (N, -1) tensor
            
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch).view(N, -1)

            # Flatten the (L, N, ...) tensors to (L * N, ...)
            share_obs_batch = _flatten_helper(L, N, share_obs_batch)
            obs_batch = _flatten_helper(L, N, obs_batch)
            actions_batch = _flatten_helper(L, N, actions_batch)
            value_preds_batch = _flatten_helper(L, N, value_preds_batch)
            return_batch = _flatten_helper(L, N, return_batch)
            masks_batch = _flatten_helper(L, N, masks_batch)
            high_masks_batch = _flatten_helper(L, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(L, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(L, N, adv_targ)
            
            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
            
