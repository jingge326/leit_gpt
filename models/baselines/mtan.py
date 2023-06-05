import time
import torch
import torch.nn as nn
import numpy as np

import utils
from experiments import utils_mtan


class MTAN(nn.Module):
    def __init__(self, args, encoder_z0, decoder):

        super(MTAN, self).__init__()

        self.args = args
        self.time_start = 0
        self.encoder_z0 = encoder_z0
        self.decoder = decoder

    def forward(self, batch, k_iwae):

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

        batch_len = data_in.shape[0]
        utils.check_mask(data_in, mask_in)

        self.time_start = time.time()
        latent = self.encoder_z0(
            torch.cat((data_in, mask_in), 2), times_in)

        qz0_mean = latent[:, :, :self.args.latent_dim]
        qz0_logvar = latent[:, :, self.args.latent_dim:]

        epsilon = torch.randn(
            k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(data_in.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])

        pred_x = self.decoder(z=z0, time_steps=times_out[None, :, :].repeat(
            k_iwae, 1, 1).view(-1, times_out.shape[1]))

        # nsample, batch, seqlen, dim
        pred_x = pred_x.view(k_iwae, batch_len,
                             pred_x.shape[1], pred_x.shape[2])

        t_loss_start = time.time()
        # compute loss
        logpx, analytic_kl = utils_mtan.compute_losses(
            data_out, mask_out, qz0_mean, qz0_logvar, pred_x, self.args)
        recon_loss = -(torch.logsumexp(logpx - self.args.kl_coef *
                       analytic_kl, dim=0).mean(0) - np.log(k_iwae))
        results = {}
        results["loss"] = recon_loss
        results['kldiv_z0'] = torch.mean(analytic_kl).detach()
        t_loss_end = time.time()
        self.time_start += t_loss_end - t_loss_start

        forward_info = {'initial_state': z0,
                        'pred_x': pred_x, }

        return results, forward_info
