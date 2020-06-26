import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_agents, num_enemies, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n
            self.base = MLPBase(obs_shape, num_agents, num_enemies, **base_kwargs)
            self.dist = Categorical(self.base.output_size, num_actions)
        elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]
            self.base = MLPBase(obs_shape, num_agents, num_enemies, **base_kwargs)
            self.dist = DiagGaussian(self.base.output_size, num_actions)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
            self.base = MLPBase(obs_shape, num_agents, num_enemies, **base_kwargs)
            self.dist = Bernoulli(self.base.output_size, num_actions)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def is_naive_recurrent(self):
        return self.base.is_naive_recurrent
        
    @property
    def is_lstm(self):
        return self.base.is_lstm
        
    @property
    def is_attn(self):
        return self.base.is_attn
        
    @property
    def is_attn_interactive(self):
        return self.base.is_attn_interactive

    @property
    def recurrent_hidden_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_size

    def forward(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
        raise NotImplementedError

    def act(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks, available_actions, deterministic=False):
        value, actor_features, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        
        dist = self.dist(actor_features, available_actions)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic

    def get_value(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        value, _, _, _,_ ,_ = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        return value

    def evaluate_actions(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks, action):
        value, actor_features, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic


class NNBase(nn.Module):
    def __init__(self, obs_shape, num_agents, num_enemies, lstm=False, naive_recurrent=False, recurrent=False, hidden_size=64, attn=False, attn_layers=1, attn_size=64, attn_head=1):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self._lstm=lstm
        self._attn=attn
                
        if self._attn:
            self.self_attn_actor = SelfAttn(obs_shape, num_agents, num_enemies, attn_layers, attn_size, attn_head)
            self.self_attn_critic = Attn(obs_shape, num_agents, attn_layers, attn_size, attn_head)
        
        assert (self._lstm and (self._recurrent or self._naive_recurrent))==False, ("LSTM and GRU can not be set True simultaneously.")

        if self._lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.lstm_critic = nn.LSTM(hidden_size, hidden_size)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            for name, param in self.lstm_critic.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        if self._recurrent or self._naive_recurrent:
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.gru_critic = nn.GRU(hidden_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            for name, param in self.gru_critic.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_naive_recurrent(self):
        return self._naive_recurrent
        
    @property
    def is_lstm(self):
        return self._lstm
        
    @property
    def is_attn(self):
        return self._attn
        
    @property
    def is_attn_interactive(self):
        return self._attn_interactive

    @property
    def recurrent_hidden_size(self):
        if self._recurrent or self._naive_recurrent or self._lstm:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)          
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)
            
            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru( x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))                  
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
        
    def _forward_gru_critic(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru_critic(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x = self.gru_critic(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru_critic(x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
        
    def _forward_lstm(self, x, hxs, c, masks):
        if x.size(0) == hxs.size(0):
            x, (hxs,c) = self.lstm(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), c.unsqueeze(0)))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)           
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            c = c.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                #rnn_scores, hxs = self.gru( x[start_idx:end_idx],hxs * masks[start_idx].view(1, -1, 1))                  
                rnn_scores, (hxs, c) = self.lstm(x[start_idx:end_idx], (hxs * masks[start_idx].view(1, -1, 1),c))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)
            c.squeeze(0)

        return x, hxs, c
        
    def _forward_lstm_critic(self, x, hxs, c, masks):
        if x.size(0) == hxs.size(0):
            x, (hxs, c) = self.lstm_critic(x.unsqueeze(0), ((hxs * masks).unsqueeze(0), c.unsqueeze(0)))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            c = c.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, (hxs,c) = self.lstm_critic(x[start_idx:end_idx], (hxs * masks[start_idx].view(1, -1, 1),c))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
            c = c.squeeze(0)

        return x, hxs, c

class MLPBase(NNBase):
    def __init__(self, obs_shape, num_agents, num_enemies, lstm = False, naive_recurrent = False, recurrent=False, hidden_size=64, attn=False, attn_layers=1, attn_size=64, attn_head=1):
        super(MLPBase, self).__init__(obs_shape, num_agents, num_enemies, lstm, naive_recurrent, recurrent, hidden_size, attn, attn_layers, attn_size, attn_head)

        if attn:
            num_inputs_actor = 3 * attn_size 
            num_inputs_critic = 2 * attn_size
        else:
            num_inputs_actor = obs_shape[0]
            num_inputs_critic = num_agents * obs_shape[0]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs_actor, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs_critic, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        share_x = share_inputs
        x = inputs
        
        if self.is_attn:
            x = self.self_attn_actor(x)
            share_x = self.self_attn_critic(agent_id, share_x)
                
        hidden_actor = self.actor(x)
        hidden_critic = self.critic(share_x)

        if self.is_recurrent or self.is_naive_recurrent:
            hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
            hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)
            
        if self.is_lstm:
            hidden_actor, rnn_hxs_actor, rnn_c_actor = self._forward_lstm(hidden_actor, rnn_hxs_actor, rnn_c_actor, masks)
            hidden_critic, rnn_hxs_critic, rnn_c_critic = self._forward_lstm_critic(hidden_critic, rnn_hxs_critic, rnn_c_critic, masks)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dout, dropout=0.1, bias=True):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Sequential(nn.Linear(n_head * d_v, n_head * d_v, bias=bias), nn.ReLU(), nn.Linear(n_head * d_v, dout, bias=bias))

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm_q = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v = nn.LayerNorm(n_head * d_v, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        residual = q

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = self.layer_norm_q(q).transpose(1, 2), self.layer_norm_k(k).transpose(1, 2), self.layer_norm_v(v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        return q, residual, attn.squeeze()
        
class MultiHeadAttention2Layers(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dout, dropout=0.1, bias=True):
        super(MultiHeadAttention2Layers, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs_1 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks_1 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs_1 = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc_1 = nn.Linear(n_head * d_v, dout, bias=bias)

        self.attention_1 = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm_q_1 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k_1 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v_1 = nn.LayerNorm(n_head * d_v, eps=1e-6)

        # 2nd layer of attention
        self.w_qs_2 = nn.Linear(n_head * d_k, n_head * d_k, bias=bias)
        self.w_ks_2 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs_2 = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc_2 = nn.Linear(n_head * d_v, dout, bias=bias)

        self.attention_2 = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm_q_2 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k_2 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v_2 = nn.LayerNorm(n_head * d_v, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #In this layer, we perform self attention
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_ = self.w_qs_1(q).view(sz_b, len_q, n_head, d_k)
        k_ = self.w_ks_1(k).view(sz_b, len_k, n_head, d_k)
        v_ = self.w_vs_1(v).view(sz_b, len_v, n_head, d_v)
        residual1 = q_

        # Transpose for attention dot product: b x n x lq x dv
        q_, k_, v_ = self.layer_norm_q_1(q_).transpose(1, 2), self.layer_norm_k_1(k_).transpose(1, 2), self.layer_norm_v_1(v_).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q_, attn1 = self.attention_1(q_, k_, v_, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_ = q_.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_ = self.fc_1(q_)

        # In second layer we use attention
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_ = self.w_qs_2(q_).view(sz_b, len_q, n_head, d_k)
        k_ = self.w_ks_2(k).view(sz_b, len_k, n_head, d_k)
        v_ = self.w_vs_2(v).view(sz_b, len_v, n_head, d_v)
        residual2 = q_

        # Transpose for attention dot product: b x n x lq x dv
        q_, k_, v_ = self.layer_norm_q_2(q_).transpose(1, 2), self.layer_norm_k_2(k_).transpose(1, 2), self.layer_norm_v_2(v_).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q_, attn2 = self.attention_2(q_, k_, v_, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_ = q_.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_ = self.fc_2(q_)
        return q_, torch.cat((residual1, residual2), dim=-1), attn2.squeeze()
        
class SelfAttn(nn.Module):
    def __init__(self, obs_shape, num_agents, num_enemies, attn_layers=1, attn_size=64, attn_head=1):
        super(SelfAttn, self).__init__()
        self.attn_layers = attn_layers
        self.attn_size = attn_size
        self.attn_head = attn_head
        self.num_agents = num_agents
        self.num_allies = num_agents-1
        self.num_enemies = num_enemies
        self.num_inputs = obs_shape[0]
        self.num_inputs_ally = obs_shape[1]
        self.num_inputs_enemy = obs_shape[2]
        self.num_inputs_own = obs_shape[3]
        self.num_inputs_move = obs_shape[4]
        self.agent_id_feats = obs_shape[5]
        self.timestep_feats = obs_shape[6]
        self.fc = nn.Linear(self.num_inputs_own + self.num_inputs_move+self.agent_id_feats+self.timestep_feats, self.attn_size)
        
        if self.attn_layers == 1:
            self.a_self_attn = MultiHeadAttention(self.attn_head, self.num_inputs_ally, self.attn_size, self.attn_size, self.attn_size)
            self.e_self_attn = MultiHeadAttention(self.attn_head, self.num_inputs_enemy, self.attn_size, self.attn_size, self.attn_size)
        elif self.attn_layers == 2:
            self.a_self_attn = MultiHeadAttention2Layers(self.attn_head, self.num_inputs_ally, self.attn_size, self.attn_size, self.attn_size)
            self.e_self_attn = MultiHeadAttention2Layers(self.attn_head, self.num_inputs_enemy, self.attn_size, self.attn_size, self.attn_size)
        
    def forward(self, inputs):
        # inputs [bs,all_size*nagents]       
        bs = inputs.size(0)
        # Own features
        if self.agent_id_feats+self.timestep_feats == 0:
            own_feats = inputs[:,-self.num_inputs_own:].view(bs, 1, -1)
        else:
            own_feats = inputs[:,-self.num_inputs_own-self.agent_id_feats-self.timestep_feats:-self.agent_id_feats-self.timestep_feats].view(bs, 1, -1)
        
        # Ally features
        ally_feats = inputs[:,0:self.num_inputs_ally*(self.num_allies)].view(bs, self.num_allies, -1)        
        ally_feats, own_feats_a, _ = self.a_self_attn(own_feats, ally_feats, ally_feats)
        #ally_own_feats = torch.cat((ally_feats.view(bs, -1), own_feats_a.view(bs, -1)), dim=-1)
        ally_own_feats = ally_feats.view(bs, -1)
        
        # Enemy features
        enemy_feats = inputs[:, self.num_inputs_ally*self.num_allies:self.num_inputs_ally*self.num_allies + self.num_inputs_enemy*self.num_enemies].view(bs, self.num_enemies, -1)
        enemy_feats, own_feats_a, _ = self.e_self_attn(own_feats[:,0:self.num_inputs_enemy], enemy_feats, enemy_feats)
        #enemy_own_feats = torch.cat((enemy_feats.view(bs, -1), own_feats_a.view(bs, -1)), dim=-1)
        enemy_own_feats = enemy_feats.view(bs, -1)
        
        #Move and own and other features
        other_feats = inputs[:, self.num_inputs_ally*self.num_allies + self.num_inputs_enemy*self.num_enemies:]
        other_feats = self.fc(other_feats)
        
        #Concat everything
        inputs = torch.cat((ally_own_feats, enemy_own_feats, other_feats), dim=-1)
        
        return inputs
        
class Attn(nn.Module):
    def __init__(self, obs_shape, num_agents, attn_layers=1, attn_size=64, attn_head=1):
        super(Attn, self).__init__()
        self.attn_layers = attn_layers
        self.attn_size = attn_size
        self.attn_head = attn_head
        self.num_agents = num_agents
        self.num_allies = num_agents-1
        
        self.num_inputs = obs_shape[0]
        self.fc = nn.Linear(self.num_inputs, self.attn_size)
        
        if self.attn_layers == 1:
            self.a_attn = MultiHeadAttention(self.attn_head, self.num_inputs, self.attn_size, self.attn_size, self.attn_size)
        elif self.attn_layers == 2:
            self.a_attn = MultiHeadAttention2Layers(self.attn_head, self.num_inputs, self.attn_size, self.attn_size, self.attn_size)
        
    def forward(self, agent_id, inputs):
        # inputs [bs,all_size*nagents]          
        bs = inputs.size(0)
        inputs = inputs.view(bs, self.num_agents, -1)
        
        # Self features
        self_obs = inputs[:,agent_id,:].view(bs, 1, -1)
        
        # Ally features
        ally_obs = inputs[:,torch.arange(self.num_agents)!=agent_id,:].view(bs, self.num_allies, -1)               
        
        ally_obs, self_obs_a, _ = self.a_attn(self_obs, ally_obs, ally_obs)
        ally_self_obs = ally_obs.view(bs, -1)
        
        self_obs = self.fc(inputs[:,agent_id,:])
        
        inputs = torch.cat((ally_self_obs, self_obs), dim=-1)        
        
        return inputs
        
    