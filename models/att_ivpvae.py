import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from experiments.utils_mtan import compute_log_normal_pdf
from models.base_models.ivpvae_components import Softmax_with_mask, Z_to_mu_ReLU, Z_to_std_ReLU

import utils


# class ITS_MultiheadAttention(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.num_heads = args.num_heads
#         self.latent_dim = args.latent_dim
#         self.mulhead_dim = self.latent_dim * self.num_heads
#         self.input_dim = args.variable_num*2 + args.embed_time

#         self.time_embedding = TimeEmbedding(args.embed_time)

#         # Linear projections for q, k, v
#         self.q_linear = nn.Linear(self.input_dim, self.mulhead_dim)
#         self.k_linear = nn.Linear(self.input_dim, self.mulhead_dim)

#         # Dropout layer
#         self.dropout = nn.Dropout(p=args.dropout)

#         # Linear projection for concatenation of heads
#         self.output_linear = nn.Linear(self.mulhead_dim, self.latent_dim)

#     def forward(self, attn_in, times_in, value, mask):

#         tt = self.time_embedding(times_in)
#         score = self.attn_lyr(torch.cat([attn_in, tt], dim=-1))

#         batch_size = x.size(0)

#         # Project input x into separate query, key, and value vectors
#         q = self.q_linear(x).view(
#             batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.k_linear(x).view(
#             batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.v_linear(x).view(
#             batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

#         # Calculate scaled dot-product attention scores
#         scores = torch.matmul(q, k.transpose(-2, -1)) / \
#             torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

#         # Mask attention scores if a mask is provided
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#             scores = scores.masked_fill(mask == 0, -1e9)

#         # Apply softmax to obtain attention weights
#         weights = nn.functional.softmax(scores, dim=-1)
#         weights = self.dropout(weights)

#         # Apply attention weights to values and concatenate across heads
#         attention_output = torch.matmul(weights, v).transpose(
#             1, 2).contiguous().view(batch_size, -1, self.mulhead_dim)

#         # Linear projection to obtain final output
#         output = self.output_linear(attention_output)

#         return output, weights


# class multiTimeAttention(nn.Module):

#     def __init__(self, input_dim, nhidden=16,
#                  embed_time=16, num_heads=1):
#         super(multiTimeAttention, self).__init__()
#         assert embed_time % num_heads == 0
#         self.embed_time = embed_time
#         self.embed_time_k = embed_time // num_heads
#         self.h = num_heads
#         self.dim = input_dim
#         self.nhidden = nhidden
#         self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
#                                       nn.Linear(embed_time, embed_time),
#                                       nn.Linear(input_dim*num_heads, nhidden)])

#     def attention(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         dim = value.size(-1)
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#             / math.sqrt(d_k)
#         # scores = torch.matmul(query, key) / math.sqrt(d_k)
#         scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
#         if mask is not None:
#             scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
#         p_attn = F.softmax(scores, dim=-2)
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#         return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

#     def forward(self, query, key, value, mask=None, dropout=None):
#         "Compute 'Scaled Dot Product Attention'"
#         batch, seq_len, dim = value.size()
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         value = value.unsqueeze(1)
#         query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
#                       for l, x in zip(self.linears, (query, key))]
#         x, _ = self.attention(query, key, value, mask, dropout)
#         x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
#         return self.linears[-1](x)

import torch
import torch.nn as nn
import torch.nn.functional as F


# class MultiheadAttention(nn.Module):
#     # MultiheadAttention for irregular time series
#     def __init__(self, input_dim, num_heads, dropout):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(input_dim, input_dim)

#     def forward(self, x, mask=None, v=None):
#         # x: shape: [batch_size, seq_len, input_dim]
#         # mask: shape: [batch_size, seq_len]
#         b, n, _ = x.size()
#         q = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(
#             1, 2)  # shape: [batch_size, num_heads, seq_len, head_dim]
#         k = self.key(x).view(b, n, self.num_heads, self.head_dim).transpose(
#             1, 2)  # shape: [batch_size, num_heads, seq_len, head_dim]
#         if v is not None:
#             v = v.view(b, n, self.num_heads, self.head_dim).transpose(
#                 1, 2)  # shape: [batch_size, num_heads, seq_len, head_dim]
#         else:
#             v = self.value(x).view(b, n, self.num_heads, self.head_dim).transpose(
#                 1, 2)  # shape: [batch_size, num_heads, seq_len, head_dim]

#         # Scaled Dot-Product Attention
#         # shape: [batch_size, num_heads, seq_len, seq_len]
#         scores = torch.matmul(q, k.transpose(-2, -1)) / \
#             torch.sqrt(torch.tensor(self.head_dim,
#                        dtype=torch.float32, device=x.device))

#         if mask is not None:
#             mask = mask.unsqueeze(1).unsqueeze(2)
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         # shape: [batch_size, num_heads, seq_len, seq_len]
#         attn_weights = F.softmax(scores, dim=-1)

#         attn_weights = self.dropout(attn_weights)

#         # shape: [batch_size, num_heads, seq_len, head_dim]
#         attn_output = torch.matmul(attn_weights, v)

#         # Concatenation and linear transformation
#         attn_output = attn_output.transpose(1, 2).contiguous().view(
#             b, n, -1)  # shape: [batch_size, seq_len, num_heads*head_dim]
#         # shape: [batch_size, seq_len, input_dim]
#         output = self.fc(attn_output)

#         return output


class MultiheadAttention(nn.Module):
    # MultiheadAttention for irregular time series
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.all_dim = input_dim * num_heads
        self.query = nn.Linear(input_dim, self.all_dim)
        self.key = nn.Linear(input_dim, self.all_dim)
        self.value = nn.Linear(input_dim, self.all_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.all_dim, input_dim)

    def forward(self, x, mask=None):
        # x: shape: [batch_size, seq_len, input_dim]
        # mask: shape: [batch_size, seq_len]
        b, n, _ = x.size()
        q = self.query(x).view(b, n, self.num_heads, self.input_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]
        k = self.key(x).view(b, n, self.num_heads, self.input_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]
        v = self.value(x).view(b, n, self.num_heads, self.input_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]

        # Scaled Dot-Product Attention
        # shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.input_dim,
                       dtype=torch.float32, device=x.device))

        if mask is not None:
            mask1 = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask1 == 0, float('-inf'))

        # shape: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # shape: [batch_size, num_heads, seq_len, input_dim]
        attn_output = torch.matmul(attn_weights, v)

        mask2 = mask.unsqueeze(1).unsqueeze(3)

        attn_output = attn_output.masked_fill(mask2 == 0, float(0))

        # Concatenation and linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            b, n, -1)  # shape: [batch_size, seq_len, num_heads*input_dim]
        # shape: [batch_size, seq_len, input_dim]
        output = self.fc(attn_output)

        return output


class AttentionReductionNet(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, num_heads, dropout)

    def forward(self, x, mask):
        # x: shape: [n_traj, batch_size, seq_len, input_dim]
        # mask: shape: [n_traj, batch_size, seq_len]
        shape_x = x.size()
        # shape: [n_traj * batch_size, seq_len, input_dim]
        x = x.view(-1, x.size(2), x.size(3))
        # shape: [n_traj * batch_size, seq_len]
        mask = mask.view(-1, mask.size(2))
        # shape: [batch_size, seq_len, input_dim]
        attended = self.attention(x, mask)

        lat_exist = mask.unsqueeze(-1)
        # shape: [batch_size, 1, input_dim]
        reduced = torch.sum(attended * lat_exist, dim=-2,
                            keepdim=True) / lat_exist.sum(dim=-2, keepdim=True)

        # reduced = torch.mean(attended, dim=1, keepdim=True)
        # shape: [n_traj, batch_size, 1, input_dim]
        x = reduced.view(shape_x[0], shape_x[1], 1, shape_x[3])
        return x


class Latent_Attention_Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)
        self.dropout = nn.Dropout(args.dropout)
        if args.combine_methods == "attn_rough":
            dim_attn = args.variable_num * 2 + args.embed_time
        else:
            dim_attn = args.latent_dim + args.embed_time

        self.attn_lyr = nn.Sequential(
            nn.Linear(dim_attn, 1),
            nn.ReLU()
        )
        utils.init_network_weights(self.periodic)
        utils.init_network_weights(self.linear)
        nn.init.kaiming_uniform_(self.attn_lyr[0].weight)

    def time_embedding(self, tt):
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def softmax_with_mask(self, input, mask, dim):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
        output = torch.exp(input) * mask
        output = output / torch.sum(output, dim=dim, keepdim=True)
        return output

    def forward(self, attn_in, times_in, lat, mask):
        tt = times_in.unsqueeze(-1)
        if len(attn_in.shape) - len(tt.shape) == 1:
            tt = tt.unsqueeze(0).repeat(attn_in.shape[0], 1, 1, 1)
        if self.args.embed_time > 1:
            tt = self.time_embedding(times_in)
        score = self.attn_lyr(torch.cat([attn_in, tt], dim=-1))
        score = self.softmax_with_mask(score, mask, dim=1)
        score = self.dropout(score)
        output = torch.sum(score * lat, dim=-2, keepdim=True)
        assert (not torch.isinf(output).any())
        assert (not torch.isnan(output).any())
        return output


class IVPVAE_Encoder(nn.Module):
    def __init__(self, latent_dim, ivp_solver, args):
        super().__init__()
        self.args = args
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
        self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, data_embeded, times_in):
        assert (not torch.isnan(data_embeded).any())
        assert (not torch.isnan(times_in).any())

        back_time_steps = torch.neg(times_in)

        latent = self.ivp_solver(data_embeded.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent) + 1e-8

        return mean_z0, std_z0


class AttIVPVAE(nn.Module):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            ivp_solver):

        super(AttIVPVAE, self).__init__()

        self.args = args
        self.time_start = 0
        self.latent_dim = args.latent_dim
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))
        # basic models
        self.embedding_nn = embedding_nn
        self.ivp_solver = ivp_solver
        self.reconst_mapper = reconst_mapper
        self.encoder_z0 = IVPVAE_Encoder(self.latent_dim, ivp_solver, args)
        # self.lat_attn = Latent_Attention_Net(args)
        self.lat_attn = AttentionReductionNet(input_dim=self.latent_dim,
                                              num_heads=self.args.nhead,
                                              dropout=self.args.dropout)

    def forward(self, batch, k_iwae=1, run_backwards=True):
        self.time_start = time.time()
        results = dict.fromkeys(
            ['likelihood', 'mse', 'forward_time', 'loss'])

        times_in = batch['times_in']
        data_in = batch['data_in']
        mask_in = batch['mask_in']
        if self.args.extrap_full == True:
            times_out = batch['times_out']
            data_out = batch['data_out']
            mask_out = batch['mask_out']
        else:
            times_out = batch['times_in']
            data_out = batch['data_in']
            mask_out = batch['mask_in']

        # Note: the start time point cannot be 0!
        t_exist = times_in.gt(torch.zeros_like(times_in))
        lat_exist = t_exist.unsqueeze(-1).repeat(1, 1, self.latent_dim)

        utils.check_mask(data_in, mask_in)
        data_embeded = self.embedding_nn(data_in, mask_in)
        assert (not torch.isnan(data_embeded).any())
        # Encoder
        first_point_mu, first_point_std = self.encoder_z0(
            data_embeded, times_in)
        first_point_mu = first_point_mu * lat_exist

        # KL Divergence Loss
        fp_distr = Normal(first_point_mu, first_point_std)
        kldiv_z0 = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        assert not torch.isinf(kldiv_z0).any()
        assert not torch.isnan(kldiv_z0).any()
        # kldiv_z0 = kldiv_z0 * lat_exist
        # kldiv_z0 = torch.mean(kldiv_z0, (1, 2))
        # mean out dimension 1, 2 of kldiv_z0 based on lat_exist
        kldiv_z0 = torch.sum(kldiv_z0 * lat_exist, (1, 2)) / \
            lat_exist.sum((1, 2))

        # Sampling
        means_z0 = first_point_mu.repeat(k_iwae, 1, 1, 1)
        sigma_z0 = first_point_std.repeat(k_iwae, 1, 1, 1)
        initial_state = utils.sample_standard_gaussian(
            means_z0, sigma_z0)

        if self.args.combine_methods == "average":
            lat_exist = lat_exist.unsqueeze(0).repeat(k_iwae, 1, 1, 1)
            initial_state = torch.sum(
                initial_state * lat_exist, dim=-2, keepdim=True) / lat_exist.sum(dim=-2, keepdim=True)
        elif self.args.combine_methods == "attn_init":
            # initial_state = self.lat_attn(
            #     initial_state, times_in, initial_state, t_exist)
            t_exist = t_exist.unsqueeze(0).repeat(k_iwae, 1, 1)
            initial_state = self.lat_attn(initial_state, t_exist)
        elif self.args.combine_methods == "attn_rough":
            initial_state = self.lat_attn(
                torch.cat([data_in, mask_in], dim=-1), times_in, initial_state, t_exist)
        else:
            raise NotImplementedError

        # Decoder
        sol_z = self.ivp_solver(
            initial_state, times_out.unsqueeze(0))

        # Reconstruction/Modeling Loss
        data_out = data_out.repeat(k_iwae, 1, 1, 1)
        mask_out = mask_out.repeat(k_iwae, 1, 1, 1)

        pred_x = self.reconst_mapper(sol_z)
        rec_likelihood = compute_log_normal_pdf(
            data_out, mask_out, pred_x, self.args)

        # IWAE loss, sum out the traj dim
        loss = -torch.logsumexp(rec_likelihood -
                                self.args.kl_coef * kldiv_z0, 0)

        assert (not torch.isnan(loss).any())
        # if torch.isnan(loss):
        #     loss = -torch.mean(rec_likelihood -
        #                        self.args.kl_coef * kldiv_z0, 0)

        results['likelihood'] = torch.mean(rec_likelihood).detach()
        # mean over the batch
        results['loss'] = torch.mean(loss, dim=0)
        results['kldiv_z0'] = torch.mean(kldiv_z0).detach()
        forward_info = {'initial_state': initial_state,
                        'sol_z': sol_z,
                        'pred_x': pred_x}

        assert not torch.isnan(results["loss"]).any()
        assert not torch.isinf(results["loss"]).any()
        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)
