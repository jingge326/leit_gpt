import time
import torch
import torch.nn as nn
from experiments.utils_metrics import masked_gaussian_log_density
from experiments.utils_mtan import compute_log_normal_pdf, mean_squared_error
from libs.stribor.net.mlp import MLP
from models.att_ivpvae import AttIVPVAE
from models.base_models.cru_losses import GaussianNegLogLik
from models.baselines.ckconv import CKCONV
from models.baselines.classic_rnn import ClassicRNN
from models.baselines.cru import CRU
from models.baselines.gru_ode_bayes import NNFOwithBayesianJumps
from models.baselines.grud import GRUD
from models.baselines.mtan import MTAN
from models.baselines.red_vae import REDVAE
from models.ivp_auto import IVPAuto
from models.ivp_vae import IVPVAE
from models.ivpvae_old import IVPVAE_OLD
from utils import log_lik_gaussian_simple


class IVPVAE_Extrap(IVPVAE):
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

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start
            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

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


class IVPVAE_OLD_Extrap(IVPVAE_OLD):
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

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start
            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

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


class IVPAuto_Extrap(IVPAuto):
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

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start
            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

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


class IVPAuto_Extrap(IVPAuto):
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

        results, forward_info = self.forward(batch)

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.unsqueeze(0)
            next_mask = mask_out.unsqueeze(0)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start

            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

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


class AttIVPVAE_Extrap(AttIVPVAE):
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

        data_out = batch['data_out']
        mask_out = batch['mask_out']

        # Forecasting
        if self.args.extrap_full == True:
            mask_extrap = batch['mask_extrap']
            pred_x = forward_info['pred_x']
            results['forward_time'] = time.time() - self.time_start
            results["mse"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
            results["mse_extrap"] = mean_squared_error(
                data_out, pred_x, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()
        else:
            sol_z = self.ivp_solver(
                forward_info['initial_state'], batch['times_out'].unsqueeze(0))

            next_data = data_out.repeat(k_iwae, 1, 1, 1)
            next_mask = mask_out.repeat(k_iwae, 1, 1, 1)
            pred_x = self.reconst_mapper(sol_z)
            results['forward_time'] = time.time() - self.time_start
            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, 0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            if self.args.train_w_reconstr:
                results["loss"] = results["loss"] + \
                    self.args.ratio_nl * loss_next
            else:
                results["loss"] = loss_next

            mse_extrap = mean_squared_error(next_data, pred_x, mask=next_mask)
            results["mse_extrap"] = torch.mean(mse_extrap).detach()
            results['mse'] = torch.mean(mean_squared_error(
                batch['data_in'], forward_info['pred_x'], mask=batch['mask_in'])).detach()
        # assert not torch.isnan(results["loss"]).any()
        # assert not torch.isinf(results["loss"]).any()
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class REDVAE_Extrap(REDVAE):
    def __init__(self, args, z0_diffeq_solver, diffeq_solver):

        super().__init__(args, z0_diffeq_solver, diffeq_solver)

        self.args = args

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        # Forecasting
        # sol_z shape [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_z = self.diffeq_solver(
            forward_info['initial_state'], batch['times_out'].unsqueeze(0))

        # Reconstruction/Modeling Loss
        next_data = batch['data_out'].repeat(k_iwae, 1, 1, 1)
        next_mask = batch['mask_out'].repeat(k_iwae, 1, 1, 1)
        pred_x = self.decoder(sol_z)
        results['forward_time'] = time.time() - self.time_start

        if self.args.fast_llloss == True:
            log_ll = compute_log_normal_pdf(
                next_data, next_mask, pred_x, self.args)
        else:
            log_ll = masked_gaussian_log_density(
                pred_x, next_data, obsrv_std=self.obsrv_std, mask=next_mask)

        # sum out the traj dim
        loss_next = -torch.logsumexp(log_ll, dim=0)
        # mean out the batch dim
        loss_next = torch.mean(loss_next, dim=0)

        mse_extrap = mean_squared_error(next_data, pred_x, mask=next_mask)

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + self.args.ratio_nl * loss_next
        else:
            loss = loss_next

        results["loss"] = loss
        results["val_loss"] = loss_next
        results["mse_extrap"] = torch.mean(mse_extrap).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class ClassicRNN_Extrap(ClassicRNN):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        if self.args.extrap_method == 'mlp':
            self.extrap_mlp = MLP(
                in_dim=self.args.latent_dim + 1,
                hidden_dims=[128, 128],
                out_dim=20,
                activation='ReLU',
                final_activation='ReLU')

    def compute_prediction_results(self, batch):
        results = dict.fromkeys(['forward_time', 'mse', 'loss'])
        forward_info = self.forward(batch)

        next_data = batch['data_out']
        mask_out = batch['mask_out']
        next_times = batch['times_out']

        last_hidden = forward_info['hidden_state']
        if self.args.extrap_method == 'mlp':
            # repeat last_hidden for all next_times and concat the time info to the last hidden state
            features = torch.cat((last_hidden.unsqueeze(1).repeat(
                1, next_times.size(1), 1), next_times.unsqueeze(-1)), dim=-1)
            hiddens = self.extrap_mlp(features)
            pred_x = self.rec_lyr(hiddens).unsqueeze(0)
        elif self.args.extrap_method == 'seq2seq':
            last_time = batch['times_in'].max(dim=-1, keepdim=True)[0]
            next_times = torch.cat((last_time, next_times), dim=1)
            delta_ts = next_times[:, 1:] - next_times[:, :-1]
            full_mask = torch.ones_like(mask_out)
            pred_x = self.generate_seq(
                first_hidden=last_hidden,
                delta_ts=delta_ts,
                cell=self.rnn_cell,
                decoder=self.rec_lyr,
                mask=full_mask)
        else:
            raise NotImplementedError

        results['forward_time'] = time.time() - self.time_start
        # Compute likelihood
        next_data = next_data.repeat(pred_x.size(0), 1, 1, 1)
        mask_out = mask_out.repeat(pred_x.size(0), 1, 1, 1)

        if self.args.fast_llloss == True:
            log_ll = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)
        else:
            log_ll = masked_gaussian_log_density(
                pred_x, next_data, obsrv_std=self.obsrv_std, mask=mask_out)

        loss_next = -torch.mean(log_ll)
        mse_extrap = mean_squared_error(next_data, pred_x, mask=mask_out)

        results["loss"] = loss_next
        results["mse_extrap"] = torch.mean(mse_extrap).detach()
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class MTAN_Extrap(MTAN):
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
            results['forward_time'] = time.time() - self.time_start
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
            results['forward_time'] = time.time() - self.time_start

            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(*next_data.shape)

            rec_likelihood = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)

            # sum out the traj dim
            loss_next = -torch.logsumexp(rec_likelihood, dim=0)
            # mean out the batch dim
            loss_next = torch.mean(loss_next, dim=0)

            assert (not torch.isnan(loss_next))

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


class CRU_Extrap(CRU):
    def __init__(self, args):
        super().__init__(args)

    def compute_prediction_results(self, batch):

        times_out = batch['times_out']
        data_out = batch['data_out']
        mask_out = batch['mask_out']
        mask_extrap = batch['mask_extrap']

        data_in = batch["data_out"] * mask_extrap.unsqueeze(-1)
        mask_in = batch["mask_out"] * mask_extrap.unsqueeze(-1)

        results = dict.fromkeys(
            ['likelihood', 'mse', 'forward_time', 'loss'])

        obs_batch = torch.cat((data_in, mask_in), -1)
        output_mean, output_var, _ = self.forward(
            obs_batch=obs_batch, time_points=times_out, obs_valid=mask_extrap)
        results['forward_time'] = time.time() - self.time_start

        loss = GaussianNegLogLik(
            data_out, output_mean, output_var, mask=mask_out)

        extrap_loss = GaussianNegLogLik(
            data_out, output_mean, output_var, mask=(~mask_extrap[..., None])*mask_out)

        results['loss'] = loss + self.args.extrap_w * extrap_loss
        # results['loss'] = loss
        results["mse"] = mean_squared_error(
            data_out, output_mean, mask=mask_out, mask_select=mask_extrap[..., None]).detach()
        results["mse_extrap"] = mean_squared_error(
            data_out, output_mean, mask=mask_out, mask_select=~mask_extrap[..., None]).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class CKCONV_Extrap(CKCONV):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.original_tts = args.next_end - args.next_start

        self.lyr_time = torch.nn.Linear(
            in_features=self.args.num_times, out_features=self.original_tts)
        # self.lyr_feature = torch.nn.Linear(
        #     in_features=self.args.no_hidden, out_features=self.args.variable_num)
        self.lyr_feature = nn.Conv1d(
            self.args.no_hidden, self.args.variable_num, kernel_size=1, stride=1, bias=False)

        nn.init.xavier_normal_(self.lyr_time.weight)
        nn.init.xavier_normal_(self.lyr_feature.weight)

    def compute_prediction_results(self, batch):
        data = batch['data']
        mask = batch['mask']
        truth = batch['truth']
        output = self.forward(data)

        pred_x = self.lyr_feature(self.lyr_time(output)).permute(0, 2, 1)
        results['forward_time'] = time.time() - self.time_start

        log_ll = compute_log_normal_pdf(
            truth, mask, pred_x, self.args)
        # mean out batch_size
        loss_next = -torch.mean(log_ll)
        assert (not torch.isnan(loss_next))
        mse_extrap = mean_squared_error(truth, pred_x, mask=mask)

        results = {}
        results["loss"] = loss_next
        results["mse_extrap"] = torch.mean(mse_extrap).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class GOB_Extrap(NNFOwithBayesianJumps):
    def __init__(self, args):

        super().__init__(args)

        self.args = args

    def compute_prediction_results(self, batch):

        _, loss, _, _, _ = self.forward(batch)

        results = {"loss": loss}

        return results

    def run_validation(self, batch):
        _, _, _, _, _, p_vec = self.forward(batch, return_path=True)

        m, v = torch.chunk(p_vec, 2, dim=-1)
        results = {}
        results['forward_time'] = time.time() - self.time_start

        z_reord, mask_reord = [], []
        val_numobs = batch['lengths_out']
        for ind in range(0, int(torch.max(val_numobs).item())):
            stepwise_mask = val_numobs > ind
            z_reord.append(batch['data_out'][stepwise_mask, ind, :])
            mask_reord.append(batch['mask_out'][stepwise_mask, ind, :])

        X_val = torch.cat(z_reord).to(self.args.device)
        M_val = torch.cat(mask_reord).to(self.args.device)

        loss = (log_lik_gaussian_simple(X_val, m, v)*M_val).sum()
        mse = (torch.pow(X_val - m, 2) * M_val).sum()/M_val.sum()

        results["mse_extrap"] = mse.detach()
        results["loss"] = loss

        return results


class GRUD_Extrap(GRUD):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        if self.args.extrap_method == 'mlp':
            self.extrap_mlp = MLP(
                in_dim=self.args.latent_dim + 1,
                hidden_dims=[128, 128],
                out_dim=20,
                activation='ReLU',
                final_activation='ReLU')

    def compute_prediction_results(self, batch):
        forward_info = self.forward(batch)

        next_data = batch['data_out']
        mask_out = batch['mask_out']
        next_times = batch['times_out']

        last_hidden = forward_info['hidden_state']
        if self.args.extrap_method == 'mlp':
            # repeat last_hidden for all next_times and concat the time info to the last hidden state
            features = torch.cat((last_hidden.unsqueeze(1).repeat(
                1, next_times.size(1), 1), next_times.unsqueeze(-1)), dim=-1)
            hiddens = self.extrap_mlp(features)
            pred_x = self.rec_lyr(hiddens).unsqueeze(0)
        elif self.args.extrap_method == 'seq2seq':
            used_times = torch.cat((batch['times_in'].max(
                dim=-1, keepdim=True)[0], next_times), dim=1)
            delta_ts = used_times[:, 1:] - used_times[:, :-1]
            pred_x = self.generate_seq(last_hidden, delta_ts=delta_ts)
        else:
            raise NotImplementedError

        results = {}
        results['forward_time'] = time.time() - self.time_start

        # Compute likelihood
        next_data = next_data.repeat(pred_x.size(0), 1, 1, 1)
        mask_out = mask_out.repeat(pred_x.size(0), 1, 1, 1)

        if self.args.fast_llloss == True:
            log_ll = compute_log_normal_pdf(
                next_data, mask_out, pred_x, self.args)
        else:
            log_ll = masked_gaussian_log_density(
                pred_x, next_data, obsrv_std=self.obsrv_std, mask=mask_out)

        loss_next = -torch.mean(log_ll)
        mse_extrap = mean_squared_error(next_data, pred_x, mask=mask_out)

        results["loss"] = loss_next
        results["mse_extrap"] = torch.mean(mse_extrap).detach()
        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
