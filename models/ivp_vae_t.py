import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from experiments.utils_mtan import compute_log_normal_pdf, mean_squared_error
from models.base_models.ivpvae_components import IVPVAE_Encoder, IVPVAE_Encoder_logstd

import utils
from experiments.utils_metrics import likelihood_data_mask, masked_gaussian_log_density, compute_mse


class IVPVAE_t(nn.Module):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            ivp_solver):

        super(IVPVAE_t, self).__init__()

        self.args = args
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))
        # basic models
        self.embedding_nn = embedding_nn
        self.ivp_solver = ivp_solver
        self.reconst_mapper = reconst_mapper
        if self.args.z0_mapper == "logstd":
            self.encoder_z0 = IVPVAE_Encoder_logstd(
                self.args.latent_dim,
                self.ivp_solver,
                self.args)
        elif self.args.z0_mapper == "softplus":
            self.encoder_z0 = IVPVAE_Encoder(
                self.args.latent_dim,
                self.ivp_solver,
                self.args)
        else:
            raise ValueError("Unknown")

    def forward(self, batch, k_iwae=1, run_backwards=True):
        start_time = time.time()

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

        results = dict.fromkeys(
            ['likelihood', 'mse', 'forward_time', 'loss_time', 'loss'])

        utils.check_mask(data_in, mask_in)
        data_embeded = self.embedding_nn(data_in, mask_in)

        # Encoder
        first_point_mu, first_point_std = self.encoder_z0(data_embeded, mask_in, times_in,
                                                          run_backwards=run_backwards)

        # Sampling
        means_z0 = first_point_mu.repeat(k_iwae, 1, 1)
        sigma_z0 = first_point_std.repeat(k_iwae, 1, 1)
        first_point_enc = utils.sample_standard_gaussian(
            means_z0, sigma_z0)
        initial_state = first_point_enc.unsqueeze(-2)

        # Decoder
        sol_z = self.ivp_solver(
            initial_state, times_out.unsqueeze(0))

        z_in = data_embeded.repeat(k_iwae, 1, 1, 1)
        zm_in = torch.ones_like(z_in)
        log_ll_z = compute_log_normal_pdf(
            z_in, zm_in, sol_z, self.args).mean(dim=1)

        # KL Divergence Loss
        fp_distr = Normal(first_point_mu, first_point_std)
        kldiv_z0 = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        if torch.isnan(kldiv_z0).any():
            print('first_point_mu: {}, first_point_std: {}'.format(
                first_point_mu, first_point_std))
            raise Exception('kldiv_z0 is Nan!')
        kldiv_z0 = torch.mean(kldiv_z0, (0, 1))

        # IWAE loss, sum out the traj dim
        loss_vae = -torch.logsumexp(log_ll_z -
                                    self.args.kl_coef * kldiv_z0, 0)
        loss_z = -torch.logsumexp(log_ll_z, 0)
        assert not torch.isnan(loss_vae)

        forward_time = time.time() - start_time
        # Embedding Loss
        pred_x = self.reconst_mapper(sol_z)
        data_out = data_out.repeat(k_iwae, 1, 1, 1)
        mask_out = mask_out.repeat(k_iwae, 1, 1, 1)
        log_ll = compute_log_normal_pdf(
            data_out, mask_out, pred_x, self.args).mean(dim=1)
        emb_loss = -torch.logsumexp(log_ll, 0)

        loss_time = time.time() - start_time - forward_time

        results['likelihood'] = torch.mean(emb_loss).detach()
        results['forward_time'] = forward_time
        results['loss_time'] = loss_time
        results['loss'] = loss_vae + emb_loss

        results['loss_vae'] = loss_vae.detach()
        results['ll_loss_z'] = loss_z.detach()
        results['ll_loss_x'] = emb_loss.detach()

        forward_info = {'first_point_mu': first_point_mu,
                        'first_point_std': first_point_std,
                        'initial_state': initial_state,
                        'sol_z': sol_z,
                        'pred_x': pred_x,
                        'kl_first_p': torch.mean(kldiv_z0).detach()}

        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)
