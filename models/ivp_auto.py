import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from experiments.utils_mtan import compute_log_normal_pdf
from models.base_models.ivpvae_components import Softmax_with_mask, Z_to_mu_ReLU, Z_to_std_ReLU

import utils


class Z0_Attention_Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)
        self.dropout = nn.Dropout(args.dropout)
        if args.combine_methods == "attn_rough":
            dim_attn = args.variable_num * 2 + args.embed_time
        elif args.combine_methods == "attn_latent":
            dim_attn = args.latent_dim + args.embed_time
        elif args.combine_methods == "attn_embed":
            dim_attn = args.latent_dim + args.embed_time

        self.attn_lyr = nn.Sequential(
            nn.Linear(dim_attn, 1),
            nn.ReLU()
        )
        self.softmax_with_mask = Softmax_with_mask(dim=1)
        utils.init_network_weights(self.periodic)
        utils.init_network_weights(self.linear)
        nn.init.kaiming_uniform_(self.attn_lyr[0].weight)

    def time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, attn_in, times_in, z0, mask):
        if self.args.embed_time == 1:
            tt = times_in.unsqueeze(-1)
        else:
            tt = self.time_embedding(times_in)
        score = self.attn_lyr(torch.cat([attn_in, tt], dim=-1))
        score = self.softmax_with_mask(score, mask)
        score = self.dropout(score)
        output = torch.sum(score * z0, dim=1, keepdim=True)
        assert (not torch.isinf(output).any())
        assert (not torch.isnan(output).any())
        return output


class IVPAuto(nn.Module):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            ivp_solver):

        super(IVPAuto, self).__init__()

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
        self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
        self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        if self.args.combine_methods != "average":
            self.z0_att_net = Z0_Attention_Net(args)

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

        utils.check_mask(data_in, mask_in)

        t_exist = times_in.gt(torch.zeros_like(times_in))
        latent_exist = t_exist.unsqueeze(-1).repeat(1, 1, self.latent_dim)

        # Embedding
        data_embeded = self.embedding_nn(data_in, mask_in)

        # Evolving backward in time
        back_time_steps = torch.neg(times_in)
        latent = self.ivp_solver(data_embeded.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        if self.args.combine_methods == "average":
            latent = torch.sum(latent*latent_exist, dim=1, keepdim=True) / \
                latent_exist.sum(dim=1, keepdim=True)
        elif self.args.combine_methods == "attn_rough":
            latent = self.z0_att_net(
                torch.cat([data_in, mask_in], dim=-1), times_in, latent, t_exist)
        elif self.args.combine_methods == "attn_latent":
            latent = self.z0_att_net(latent, times_in, latent, t_exist)
        elif self.args.combine_methods == "attn_embed":
            latent = self.z0_att_net(
                data_embeded, times_in, latent, t_exist)
        else:
            raise NotImplementedError

        # Sampling
        z0_mean = self.z2mu_mapper(latent)
        z0_std = self.z2std_mapper(latent) + 1e-8
        z0_mean_iwae = z0_mean.repeat(k_iwae, 1, 1, 1)
        z0_std_iwae = z0_std.repeat(k_iwae, 1, 1, 1)
        initial_state = utils.sample_standard_gaussian(
            z0_mean_iwae, z0_std_iwae)

        # Evolving forward along time
        sol_z = self.ivp_solver(
            initial_state, times_out.unsqueeze(0))

        # Reconstruction
        if self.args.decoupled_ae == True:
            pred_x = self.reconst_mapper(data_embeded)
        else:
            pred_x = self.reconst_mapper(sol_z)

        ############################
        # Compute Losses

        # KL Divergence Loss
        fp_distr = Normal(z0_mean, z0_std)
        kldiv_z0 = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        # Mean over the time and latent dimension
        kldiv_z0 = torch.mean(kldiv_z0, dim=(1, 2))

        # Reconstruction loss (Loglikelihood) of Z
        ll_z = compute_log_normal_pdf(
            data_embeded, latent_exist, sol_z, self.args)

        # ELBO of Z
        # sum out the traj dim
        loss_vae = -torch.logsumexp(ll_z - self.args.kl_coef * kldiv_z0, 0)

        # Reconstruction loss (Loglikelihood) of X-autoencoder
        data_out = data_out.repeat(k_iwae, 1, 1, 1)
        mask_out = mask_out.repeat(k_iwae, 1, 1, 1)
        ll_x = compute_log_normal_pdf(data_out, mask_out, pred_x, self.args)
        # sum out the traj dim
        loss_ae = -torch.logsumexp(ll_x, 0)

        # The overall loss
        loss = loss_vae + self.args.ratio_ae*loss_ae
        # mean over the batch
        loss = torch.mean(loss, dim=0)

        assert not (torch.isnan(loss)).any()
        assert (not torch.isinf(loss).any())
        results["loss"] = loss

        results['kldiv_z0'] = torch.mean(kldiv_z0).detach()
        results['loss_ae'] = torch.mean(loss_ae).detach()
        results['loss_vae'] = torch.mean(loss_vae).detach()
        forward_info = {'initial_state': initial_state,
                        'sol_z': sol_z,
                        'pred_x': pred_x}

        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)
