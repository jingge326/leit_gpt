import torch
from experiments.utils_metrics import masked_gaussian_log_density
from experiments.utils_mtan import compute_log_normal_pdf, mean_squared_error
from models.base_models.cru_losses import GaussianNegLogLik, mse
from models.base_models.mtan_components import create_classifier
from models.baselines.classic_rnn import ClassicRNN
from models.baselines.cru import CRU
from models.baselines.mtan import MTAN
from models.baselines.red_vae import REDVAE
from models.base_models.rnn_utils import generate_seq, run_rnn
from models.gpts import GPTS
from models.ivp_vae import IVPVAE
from models.ivp_vae_t import IVPVAE_t


class GPTS_Interp(GPTS):
    def __init__(self, args):
        super().__init__(args)

    def compute_prediction_results(self, batch):
        results = self.forward(batch)
        times_out = batch['times_out']
        mask_gen = torch.ones_like(batch['mask_out'])

        exist_edge = torch.logical_xor(torch.concat([batch['exist_times'][:, 1:], torch.zeros(
            batch['exist_times'].shape[0], 1).to(batch["data_in"])], dim=-1), batch['exist_times'])

        self.zero_delta_t = self.zero_tensor.repeat(
            batch['times_in'].size(0), 1)
        delta_ts = batch['times_in'][:, 1:] - batch['times_in'][:, :-1]
        delta_ts = torch.cat((delta_ts, self.zero_delta_t), dim=1)
        ts_new = exist_edge * batch["times_out"][:, [0]]
        delta_ts = delta_ts + ts_new

        # evolved_states = self.evolve(results["latent_states"], delta_ts)

        latent_states = results["latent_states"]
        B, T, C = latent_states.size()
        latent_states = latent_states.view(
            B, T, self.args.nhead, C // self.args.nhead).unsqueeze(-2)
        delta_ts = delta_ts.view(B, T, 1, 1).repeat(1, 1, self.args.nhead, 1)

        evolved_states = self.evolve(latent_states, delta_ts).view(B, T, C)

        pred_x = torch.zeros_like(batch['data_out'])

        pred_x[:, 0, :] = (self.lm_head(evolved_states) *
                           exist_edge.unsqueeze(-1)).sum(dim=-2)

        for i in range(times_out.size(-1) - 1):
            delta_ts = times_out[:, 1:i+2] - times_out[:, :i+1]
            time_embed = self.time_embedding(times_out[:, :i+1].unsqueeze(-1))
            x = torch.cat(
                (pred_x[:, :i+1, :], mask_gen[:, :i+1, :], time_embed), dim=-1)
            x = self.input_lyr(x)
            x = self.multi_attn_lyrs(x)
            x = self.ln_f(x)

            B, T, C = x.size()
            x = x.view(B, T, self.args.nhead, C //
                       self.args.nhead).unsqueeze(-2)
            delta_ts = delta_ts.view(B, T, 1, 1).repeat(
                1, 1, self.args.nhead, 1)

            x = self.evolve(x, delta_ts).view(B, T, C)

            pred_x[:, i+1, :] = self.lm_head(x)[:, -1, :]

        rec_likelihood = compute_log_normal_pdf(
            batch['data_out'], batch['mask_out'], pred_x, self.args)
        results["loss"] = -rec_likelihood.mean()

        results["mse"] = mean_squared_error(
            batch['data_out'], pred_x, mask=batch['mask_out']).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class IVPVAE_Interp(IVPVAE_t):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        self.args = args

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        results['mse'] = torch.mean(mean_squared_error(
            batch['data_out'], forward_info['pred_x'], mask=batch['mask_out'])).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class REDVAE_Interp(REDVAE):
    def __init__(self, args, z0_diffeq_solver, diffeq_solver):

        super().__init__(args, z0_diffeq_solver, diffeq_solver)

        self.args = args

    def compute_prediction_results(self, batch, k_iwae=1):

        results, _ = self.forward(batch, k_iwae)
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class ClassicRNN_Interp(ClassicRNN):
    def __init__(self, args):

        super().__init__(args)
        self.args = args

    def compute_prediction_results(self, batch):
        results, _ = self.forward(batch)
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class MTAN_Interp(MTAN):
    def __init__(self, args, encoder_z0, decoder):
        super().__init__(args, encoder_z0, decoder)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)
        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            times_out = batch['times_out']
            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.decoder(z=forward_info['initial_state'], time_steps=times_out[None, :, :].repeat(
                k_iwae, 1, 1).view(-1, times_out.shape[1]))

            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(*next_data.shape)

            log_ll = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)
            # mean out traj dim
            rec_likelihood = torch.mean(log_ll, 0)
            # sum out the batch dim
            loss_next = -torch.logsumexp(rec_likelihood, 0)
            assert not torch.isnan(loss_next)

            if self.args.train_w_reconstr:
                results["loss"] = results["loss"] + \
                    self.args.ratio_nl * loss_next
            else:
                results["loss"] = loss_next

            mse_extrap = mean_squared_error(next_data, pred_x, mask=next_mask)
            results["mse_extrap"] = torch.mean(mse_extrap).detach()
            results['mse'] = torch.mean(mean_squared_error(
                batch['data_in'], forward_info['pred_x'], mask=batch['mask_in'])).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class CRU_Interp(CRU):
    def __init__(self, args):
        super().__init__(args)

    def compute_prediction_results(self, batch):

        times_out = batch['times_out']
        data_out = batch['data_out']
        mask_out = batch['mask_out']
        mask_extrap = batch['mask_extrap']

        results = dict.fromkeys(
            ['likelihood', 'mse', 'forward_time', 'loss'])

        output_mean, output_var, _ = self.forward(
            obs_batch=data_out, time_points=times_out, obs_valid=mask_extrap)

        loss = GaussianNegLogLik(
            data_out, output_mean, output_var, mask=mask_out)

        extrap_loss = GaussianNegLogLik(
            data_out, output_mean, output_var, mask=(~mask_extrap[..., None])*mask_out)

        results['loss'] = loss + 10 * extrap_loss
        # results['loss'] = loss
        results["mse"] = mean_squared_error(
            data_out, output_mean, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
        results["mse_extrap"] = mean_squared_error(
            data_out, output_mean, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
