import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from experiments.utils_mtan import compute_log_normal_pdf, mean_squared_error

import utils
from experiments.utils_metrics import masked_gaussian_log_density


def get_mask(x):
    x = x.unsqueeze(0)
    n_data_dims = x.size(-1)//2
    mask = x[:, :, n_data_dims:]
    utils.check_mask(x[:, :, :n_data_dims], mask)
    mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
    assert (not torch.isnan(mask).any())
    return mask.squeeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None, z0_dim=None,
                 device=torch.device('cpu')):
        super().__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        self.lstm = nn.LSTMCell(input_dim, latent_dim)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2),)
        utils.init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        latent = self.run_odernn(data, time_steps, run_backwards)

        # latent = latent.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)
            h = self.z0_diffeq_solver(h.unsqueeze(1), t).squeeze(1)

            xi = data[:, i, :]
            h_, c_ = self.lstm(xi, (h, c))
            mask = get_mask(xi)

            h = mask * h_ + (1 - mask) * h
            c = mask * c_ + (1 - mask) * c

            prev_t, t_i = time_steps[:, i], time_steps[:, i-1]

        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        decoder = nn.Sequential(nn.Linear(latent_dim, input_dim),)
        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


class REDVAE(nn.Module):
    def __init__(self, args, z0_diffeq_solver, diffeq_solver):
        super(REDVAE, self).__init__()
        self.args = args
        self.time_start = 0
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))
        # basic models
        self.encoder_z0 = Encoder_z0_ODE_RNN(
            self.args.encoder_dim,
            self.args.variable_num * 2,
            z0_diffeq_solver,
            z0_dim=self.args.latent_dim)
        self.diffeq_solver = diffeq_solver
        self.decoder = Decoder(self.args.latent_dim,
                               self.args.variable_num)

    def forward(self, batch, k_iwae, run_backwards=True):

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

        self.time_start = time.time()

        utils.check_mask(data_in, mask_in)
        data_w_mask = torch.cat((data_in, mask_in), -1)
        first_point_mu, first_point_std = self.encoder_z0(data_w_mask, times_in,
                                                          run_backwards=run_backwards)

        first_point_mu = first_point_mu.unsqueeze(1)
        first_point_std = first_point_std.unsqueeze(1)

        means_z0 = first_point_mu.repeat(k_iwae, 1, 1, 1)
        sigma_z0 = first_point_std.repeat(k_iwae, 1, 1, 1)
        initial_state = utils.sample_standard_gaussian(
            means_z0, sigma_z0)

        # sol_z shape [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_z = self.diffeq_solver(
            initial_state, times_out.unsqueeze(0))
        pred_x = self.decoder(sol_z)

        t_loss_start = time.time()
        # KL Divergence Loss
        fp_distr = Normal(first_point_mu, first_point_std)
        kldiv_z0 = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        if torch.isnan(kldiv_z0).any():
            print('first_point_mu: {}, first_point_std: {}'.format(
                first_point_mu, first_point_std))
            raise Exception('kldiv_z0 is Nan!')
        kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

        # Reconstruction/Modeling Loss
        data_out = data_out.repeat(pred_x.size(0), 1, 1, 1)
        mask_out = mask_out.repeat(pred_x.size(0), 1, 1, 1)

        if self.args.fast_llloss == True:
            rec_likelihood = compute_log_normal_pdf(
                data_out, mask_out, pred_x, self.args)
        else:
            rec_likelihood = masked_gaussian_log_density(
                pred_x, data_out, obsrv_std=self.obsrv_std, mask=mask_out)

        # sum out the traj dim
        loss = -torch.logsumexp(rec_likelihood -
                                self.args.kl_coef * kldiv_z0, 0)
        # mean over the batch
        loss = torch.mean(loss, dim=0)
        mse = mean_squared_error(data_out, pred_x, mask=mask_out)
        results['mse'] = torch.mean(mse).detach()
        results['likelihood'] = torch.mean(rec_likelihood).detach()
        results['kldiv_z0'] = torch.mean(kldiv_z0).detach()
        results['loss'] = loss
        t_loss_end = time.time()
        self.time_start += t_loss_end - t_loss_start

        forward_info = {'first_point_mu': first_point_mu,
                        'first_point_std': first_point_std,
                        'initial_state': initial_state,
                        'sol_z': sol_z}

        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)
