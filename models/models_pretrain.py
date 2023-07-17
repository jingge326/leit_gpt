import time
import torch
import torch.nn as nn
from models.bert import BERT
from models.gpts import GPTS

from experiments.utils_mtan import compute_log_normal_pdf, mean_squared_error


class GPTS_PreTrain(GPTS):
    def __init__(self, args):
        super().__init__(args)

    def compute_prediction_results(self, batch):
        results = self.forward(batch)
        times = batch['times_in']

        self.zero_delta_t = self.zero_tensor.repeat(times.size(0), 1)
        delta_ts = times[:, 1:] - times[:, :-1]
        delta_ts = torch.cat((delta_ts, self.zero_delta_t), dim=1)

        latent_states = results["latent_states"]
        B, T, C = latent_states.size()
        latent_states = latent_states.view(
            B, T, self.args.nhead, C // self.args.nhead).unsqueeze(-2)
        delta_ts = delta_ts.view(B, T, 1, 1).repeat(1, 1, self.args.nhead, 1)

        evolved_states = self.evolve(latent_states, delta_ts).view(B, T, C)

        results["gen_values"] = self.lm_head(evolved_states)
        results['forward_time'] = time.time() - self.time_start

        rec_likelihood = compute_log_normal_pdf(
            batch['data_in'][:, 1:, :], batch['mask_in'][:, 1:, :], results["gen_values"][:, :-1, :], self.args)
        results["loss"] = -rec_likelihood.mean()

        results["mse"] = mean_squared_error(
            batch['data_in'][:, 1:, :], results["gen_values"][:, :-1, :], mask=batch['mask_in'][:, 1:, :]).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class BERT_PreTrain(BERT):
    def __init__(self, args):
        super().__init__(args)

    def compute_prediction_results(self, batch):
        results = self.forward(batch)

        times_out = batch['times_out']
        times_in = batch['times_in']
        exist_times_in = batch["exist_times_in"]
        delta_ts = times_out.unsqueeze(-2) - times_in.unsqueeze(-1)

        latent_states = results["latent_states"].unsqueeze(-2)

        B, ST, _, C = latent_states.size()
        TT = delta_ts.size(-1)
        latent_states = latent_states.view(
            B, ST, 1, self.args.nhead, C // self.args.nhead).permute(0, 1, 3, 2, 4)

        evolved_states = self.evolve(
            latent_states, delta_ts.unsqueeze(-2).repeat(1, 1, self.args.nhead, 1)).view(B, ST, TT, C)

        exist_times_in = exist_times_in.view(B, ST, 1, 1)
        evolved_states = (
            evolved_states * exist_times_in).sum(dim=1)/exist_times_in.sum(dim=1)

        pred_x = self.lm_head(evolved_states)

        rec_likelihood = compute_log_normal_pdf(
            batch['data_out'], batch['mask_out'], pred_x, self.args)
        results["loss"] = -rec_likelihood.mean()

        results["mse_interp"] = mean_squared_error(
            batch['data_out'], pred_x, mask=batch['mask_out']).detach()
        results["mse"] = results["mse_interp"]

        # detect if results["loss"] is nan
        assert results["loss"].isnan() == False

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
