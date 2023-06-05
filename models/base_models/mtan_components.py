import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ivp_solvers.flow import CouplingFlow, ResNetFlow
from models.ivp_solvers.gru import GRUFlow
from models.ivp_solvers.ode import ODEModel

import utils


class create_classifier(nn.Module):

    def __init__(self, latent_dim, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, N))

    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))


class multiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        # scores = torch.matmul(query, key) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, latent_dim, num_ref_points, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=False):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.nhidden = nhidden
        self.register_buffer('query', torch.linspace(0, 1., num_ref_points))
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(
            2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(
            nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out1 = self.linear(tt)
        out2 = torch.sin(self.periodic(tt))
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class SolverWrapper(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, x, t, backwards=False):
        x = x.unsqueeze(-2)
        t = t.unsqueeze(-1).unsqueeze(2)
        if t.shape[-3] != x.shape[-3]:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4:
            t = t.repeat_interleave(x.shape[0], dim=0)
        # x: Initial conditions, (..., 1, dim)
        # t: Times to solve at, (..., seq_len, dim)
        y = self.solver(x, t)
        # y: (..., seq_len, dim)
        return y


class enc_mtan_ivp(nn.Module):
    def __init__(self, args):
        super(enc_mtan_ivp, self).__init__()
        self.args = args
        self.inpu_dim = args.variable_num
        # self.hiddens_to_z0 = nn.Sequential(
        #     nn.Linear(2*self.args.nhidden, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, self.args.latent_dim * 2))
        self.q_w = nn.Sequential(
            nn.Linear(self.inpu_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, self.args.embed_time))
        self.v_w = nn.Sequential(
            nn.Linear(self.inpu_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, self.args.embed_time))
        utils.init_network_weights(self.q_w, 'kaiming_uniform_')
        utils.init_network_weights(self.v_w, 'kaiming_uniform_')
        self.ivp_solver = self._build_ivp_solvers()

    def _build_ivp_solvers(self):
        ivp_solver = None
        hidden_dims = [self.args.hidden_dim] * self.args.hidden_layers
        if self.args.ivp_solver == 'ode':
            ivp_solver = SolverWrapper(ODEModel(self.args.latent_dim, self.args.odenet, hidden_dims, self.args.activation,
                                                self.args.final_activation, self.args.ode_solver, self.args.solver_step, self.args.atol, self.args.rtol))
        else:
            if self.args.ivp_solver == 'couplingflow':
                flow = CouplingFlow
            elif self.args.ivp_solver == 'resnetflow':
                flow = ResNetFlow
            elif self.args.ivp_solver == 'gruflow':
                flow = GRUFlow
            else:
                raise NotImplementedError

            ivp_solver = SolverWrapper(flow(
                self.args.latent_dim, self.args.flow_layers, hidden_dims, self.args.time_net, self.args.time_hidden_dim))

        return ivp_solver

    def evolve_qk(self, query, time_steps):
        qk = self.ivp_solver(query, time_steps)
        return qk

    def forward(self, x, time_steps):
        query = self.q_w(x)
        value = self.v_w(x)
        qk = self.evolve_qk(query, time_steps)
        out = qk*value
        out = self.hiddens_to_z0(out)
        return out


class dec_mtan_rnn(nn.Module):

    def __init__(self, input_dim, latent_dim, num_ref_points, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=False):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.nhidden = nhidden
        self.register_buffer('query', torch.linspace(0, 1., num_ref_points))
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(
            2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden,
                              bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps)
            key = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            query = self.fixed_time_embedding(time_steps)
            key = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out
