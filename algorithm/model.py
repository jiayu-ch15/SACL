import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.util import init
import copy
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_agents, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n
            self.base = MLPBase(obs_shape, num_agents, **base_kwargs)
            self.dist = Categorical(self.base.output_size, num_actions)
        elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]
            self.base = MLPBase(obs_shape, num_agents, **base_kwargs)
            self.dist = DiagGaussian(self.base.output_size, num_actions)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
            self.base = MLPBase(obs_shape, num_agents, **base_kwargs)
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
    def __init__(self, obs_shape, num_agents, lstm=False, naive_recurrent=False, recurrent=False, hidden_size=64, attn=False, attn_size=512, attn_N=2, attn_heads=8, average_pool=True):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self._lstm=lstm
        self._attn=attn
                
        if self._attn:
            self.encoder_actor = Encoder(obs_shape[0], obs_shape[1:], attn_size, attn_N, attn_heads, average_pool)
            self.encoder_critic = Encoder(obs_shape[0]*num_agents, [[1,obs_shape[0]]]*num_agents, attn_size, attn_N, attn_heads, average_pool)
        
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
    def __init__(self, obs_shape, num_agents, lstm = False, naive_recurrent = False, recurrent=False, hidden_size=64, attn=False, attn_size=512, attn_N=2, attn_heads=8, average_pool=True):
        super(MLPBase, self).__init__(obs_shape, num_agents, lstm, naive_recurrent, recurrent, hidden_size, attn, attn_size, attn_N, attn_heads, average_pool)

        if attn:           
            if average_pool == True:
                num_inputs_actor = attn_size + obs_shape[-1][1]
                num_inputs_critic = attn_size + obs_shape[0]
            else:
                num_inputs = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    num_inputs += split_shape[i][0]
                num_inputs_actor = num_inputs * attn_size
                num_inputs_critic = num_agents * attn_size
        else:
            num_inputs_actor = obs_shape[0]
            num_inputs_critic = obs_shape[0]*num_agents
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
            x = self.encoder_actor(x)
            share_x = self.encoder_critic(share_x, agent_id)
                
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
        super(FeedForward, self).__init__() 
        # We set d_ff as a default to 2048
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
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
        super(MultiHeadAttention, self).__init__()
        
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0)) 
        
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
        super(EncoderLayer, self).__init__()
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

# [L,[1,2],[1,2],[1,2]]   
def split_obs(obs, split_shape):
    start_idx = 0
    split_obs = []
    for i in range(len(split_shape)):
        split_obs.append(obs[:,start_idx:(start_idx+split_shape[i][0]*split_shape[i][1])])
        start_idx += split_shape[i][0]*split_shape[i][1]
    return split_obs
    
class SelfEmbedding(nn.Module):
    def __init__(self, split_shape, d_model):
        super(Embedding, self).__init__()
        self.split_shape = split_shape
                
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))

        for i in range(len(split_shape)):
            if i==(len(split_shape)-1):
                setattr(self,'fc_'+str(i), init_(nn.Linear(split_shape[i][1], d_model)))
            else:
                setattr(self,'fc_'+str(i), init_(nn.Linear(split_shape[i][1]+split_shape[-1][1], d_model)))
                  
        
    def forward(self, x, self_idx=-1):
        x = split_obs(x,self.split_shape)
        N = len(x)
        
        x1 = []  
        self_x = x[self_idx]      
        for i in range(N-1):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                temp = torch.cat((x[i][:,L*j:(L*j+L)],self_x),dim=-1)
                exec('x1.append(self.fc_{}(temp))'.format(i))
        temp = x[self_idx]
        exec('x1.append(self.fc_{}(temp))'.format(N-1))

        out = torch.stack(x1,1)        
                            
        return out, self_x
        
class Embedding(nn.Module):
    def __init__(self, split_shape, d_model):
        super(Embedding, self).__init__()
        self.split_shape = split_shape
                
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))

        for i in range(len(split_shape)):
            setattr(self,'fc_'+str(i), init_(nn.Linear(split_shape[i][1], d_model)))
                  
        
    def forward(self, x, self_idx):
        x = split_obs(x,self.split_shape)
        N = len(x)
        
        x1 = []   
        self_x = x[self_idx]     
        for i in range(N):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                temp = x[i][:,L*j:(L*j+L)]
                exec('x1.append(self.fc_{}(temp))'.format(i))

        out = torch.stack(x1,1)        
                            
        return out, self_x
    
class Encoder(nn.Module):
    def __init__(self, input_size, split_shape=None, d_model=512, attn_N=2, heads=8, average_pool=True):
        super(Encoder, self).__init__()
        
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
        self.attn_N = attn_N
        self.average_pool = average_pool
        if split_shape[0].__class__ == list:
            self.embedding = Embedding(split_shape, d_model)
        else:
            self.embedding = SelfEmbedding(split_shape, d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), self.attn_N)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, self_idx=-1, mask=None):
        x, self_x = self.embedding(src, self_idx)
        for i in range(self.attn_N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        if self.average_pool:
            x = torch.transpose(x,1,2) 
            x = F.avg_pool1d(x, kernel_size=x.size(-1)).view(x.size(0),-1)
            x = torch.cat((x, self_x),dim=-1)
        x = x.view(x.size(0),-1)
        return x    
    
