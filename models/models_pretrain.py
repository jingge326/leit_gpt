import time
import torch
import torch.nn as nn
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
