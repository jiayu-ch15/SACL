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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        init_ = lambda m: nn.init.xavier_uniform_(m)
        self.linear_1 = init_(nn.Linear(d_model, d_ff))
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        init_ = lambda m: nn.init.xavier_uniform_(m)
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = init_(nn.Linear(d_model, d_model))
        self.v_linear = init_(nn.Linear(d_model, d_model))
        self.k_linear = init_(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(d_model, d_model))
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
        
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class Encoder(nn.Module):
    def __init__(self, input_size, d_model=512, attn_N=2, heads=8):
        super(Encoder, self).__init__()
        
        init_ = lambda m: nn.init.xavier_uniform_(m)
        
        self.attn_N = attn_N
        self.fc = init_(nn.Linear(input_size, d_model))
        self.layers = get_clones(EncoderLayer(d_model, heads), self.attn_N)
        self.norm = nn.layerNorm(d_model)
        
    def forward(self, src, mask=None):
        x = self.fc(src)
        for i in range(self.attn_N):
            x = self.layers[i](x, mask)
        return self.norm(x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
class SelfAttn(nn.Module):
    def __init__(self, obs_shape, num_agents, num_enemies, attn_layers=1, attn_size=64, attn_heads=1):
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
        
    