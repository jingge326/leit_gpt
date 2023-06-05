import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import utils
from experiments.utils_metrics import masked_gaussian_log_density, compute_mse


class MTANIVP(nn.Module):
    def __init__(
            self,
            args,
            encoder_z0,
            reconst_mapper,
            diffeq_solver):

        super(MTANIVP, self).__init__()

        self.args = args
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))
        # basic models
        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.reconst_mapper = reconst_mapper
        self.com_mean = nn.Linear(self.args.embed_time, 1)
        self.com_sigma = nn.Sequential(
            nn.Linear(self.args.embed_time, 1),
            nn.Softplus())
        utils.init_network_weights(self.com_mean)
        utils.init_network_weights(self.com_sigma)

    def get_reconstruction(self, time_steps_to_predict, data, data_time_steps,
                           mask=None, n_traj_samples=1, run_backwards=True):

        latent = self.encoder_z0(
            torch.cat((data, mask), 2), data_time_steps)

        first_point_mu = self.com_mean(
            latent[:, :, :self.args.latent_dim].permute(0, 2, 1)).squeeze()
        first_point_std = self.com_sigma(
            latent[:, :, self.args.latent_dim:].permute(0, 2, 1)).squeeze()

        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)

        print('Note! n_traj_samples!')
        print(n_traj_samples)

        first_point_enc = utils.sample_standard_gaussian(
            means_z0, sigma_z0)

        # sol_z shape [n_traj_samples, n_samples, n_timepoints, n_latents]
        initial_state = first_point_enc.unsqueeze(-2)

        sol_z = self.diffeq_solver(
            initial_state, time_steps_to_predict.unsqueeze(0))

        pred_x = self.reconst_mapper(sol_z)

        forward_info = {'first_point_mu': first_point_mu,
                        'first_point_std': first_point_std,
                        'initial_state': initial_state,
                        'sol_z': sol_z}

        return pred_x, forward_info

    def forward(self, batch):
        # Condition on subsampled points
        # Make predictions for all the points
        mask = batch["mask"]
        pred_x, forward_info = self.get_reconstruction(
            time_steps_to_predict=batch["times"],
            data=batch["data"],
            data_time_steps=batch["times"],
            mask=mask,
            n_traj_samples=self.args.k_iwae)

        # print('get_reconstruction done-- computing likelihood')
        fp_mu, fp_std = forward_info['first_point_mu'], forward_info['first_point_std']
        fp_distr = Normal(fp_mu, fp_std)

        assert torch.sum(fp_std < 0) == 0.0

        kldiv_z0 = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))

        if torch.isnan(kldiv_z0).any():
            print("fp_mu: {}, fp_std: {}".format(fp_mu, fp_std))
            raise Exception("kldiv_z0 is Nan!")

        kldiv_z0 = torch.mean(kldiv_z0, (0, 1))

        data = batch["data"].repeat(pred_x.size(0), 1, 1, 1)
        mask = mask.repeat(pred_x.size(0), 1, 1, 1)

        log_ll = masked_gaussian_log_density(
            pred_x, data, obsrv_std=self.obsrv_std, mask=mask)

        # shape: [n_traj_samples]
        rec_likelihood = torch.mean(log_ll.permute(1, 0), 1)

        # IWAE loss
        loss = -torch.logsumexp(rec_likelihood -
                                self.args.kl_coef * kldiv_z0, 0)
        if torch.isnan(loss):
            loss = -torch.mean(rec_likelihood -
                               self.args.kl_coef * kldiv_z0, 0)

        mse = compute_mse(pred_x, data, mask=mask)    # shape: [1]

        results = {}
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["loss"] = loss
        forward_info["kl_first_p"] = torch.mean(kldiv_z0).detach()
        forward_info["std_first_p"] = torch.mean(fp_std).detach()

        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch)
