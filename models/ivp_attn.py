import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from libs.stribor import net

from models.ivp_solvers import CouplingFlow, ODEModel, ResNetFlow, GRUFlow
import utils


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResNetFlowBlock(nn.Module):
    def __init__(self, dim, hidden_dims, activation='ReLU', final_activation=None, time_net=None,
                 time_hidden_dim=None, n_power_iterations=5, invertible=True, **kwargs):
        super().__init__()

        self.invertible = invertible
        wrapper = None

        if invertible:
            def wrapper(layer): return torch.nn.utils.spectral_norm(
                layer, n_power_iterations=n_power_iterations)
        self.net = net.MLP(dim + 1, hidden_dims, dim,
                           activation, final_activation, wrapper_func=wrapper)
        self.time_net = getattr(net, time_net)(
            dim, hidden_dim=time_hidden_dim)

    def forward(self, x, t):
        return x + self.time_net(t) * self.net(torch.cat([x, t], -1))

    def inverse(self, y, t, iterations=100):
        if not self.invertible:
            raise NotImplementedError
        # fixed-point iteration
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.net(torch.cat([x, t], -1))
            x = y - residual
        return x


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)

        mask_left_attn = torch.tril(torch.ones(
            T, T)).view(1, 1, T, T).to(x.device)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask_left_attn == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalIVPAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.resnet_flow = ResNetFlowBlock(
            dim=config.n_embd//config.nhead,
            hidden_dims=[config.hidden_dim] * config.hidden_layers,
            time_net=config.time_net,
            time_hidden_dim=config.time_hidden_dim)

    def forward(self, x, t):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)

        mask_left_attn = torch.tril(torch.ones(
            T, T)).view(1, 1, T, T).to(x.device)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask_left_attn == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        t0 = t.reshape(t.shape[0], 1, t.shape[1], 1).repeat(
            1, self.nhead, 1, 1)
        t1 = t.reshape(t.shape[0], 1, 1, t.shape[1]).repeat(
            1, self.nhead, 1, 1)
        tt = (t1 - t0).unsqueeze(-1)
        vv = v.unsqueeze(-2).repeat_interleave(tt.shape[-2], dim=-2)
        vt = self.resnet_flow(vv, tt)
        vt = vt.transpose(-2, -3)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # (B, nh, T, T, 1) x (B, nh, T, T, hs) -> (B, nh, T, T, hs)
        y = att.unsqueeze(-1) * vt
        y = y.sum(dim=-2)  # (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.attn_types == "ivp_causal":
            self.attn = CausalIVPAttention(config)
        elif config.attn_types == "vanilla":
            self.attn = SelfAttention(config)
        elif config.attn_types == "causal":
            self.attn = CausalSelfAttention(config)
        else:
            raise ValueError("Unsupported training objective")
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, xt):
        if isinstance(xt, tuple):
            x, t = xt
            x = x + self.attn(self.ln_1(x), t)
        else:
            x = xt
            x = x + self.attn(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)
        utils.init_network_weights(self.periodic)
        utils.init_network_weights(self.linear)

    def forward(self, tt):
        tt = torch.log(tt + 1e-5)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


class IVPAttn(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.multi_attn_lyrs = nn.Sequential(
            *[Block(args) for _ in range(args.mhatt_n_layer)])
        self.ln_f = LayerNorm(args.n_embd, bias=args.bias)
        self.lm_head = nn.Linear(args.n_embd, args.variable_num)
        if args.attn_types == "ivp_causal":
            self.input_lyr = nn.Linear(
                args.variable_num, args.n_embd)
        else:
            self.input_lyr = nn.Linear(
                args.variable_num + args.embed_time, args.n_embd)
            self.time_embedding = TimeEmbedding(args)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * args.mhatt_n_layer))

        # report number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (self.num_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch):
        results = {}
        self.time_start = time.time()
        times_in = batch['times_in']
        data_in = batch['data_in'].unsqueeze(-1)

        if self.args.attn_types == "ivp_causal":
            x = self.input_lyr(data_in)
            x = self.multi_attn_lyrs((x, times_in))
        else:
            time_embed = self.time_embedding(times_in.unsqueeze(-1))
            x = torch.cat((data_in, time_embed), dim=-1)
            x = self.input_lyr(x)
            x = self.multi_attn_lyrs(x)

        x = self.ln_f(x)

        results["latent_states"] = x

        return results
