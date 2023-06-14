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
        results["gen_values"] = self.lm_head(results["latent_states"])
        results['forward_time'] = time.time() - self.time_start

        rec_likelihood = compute_log_normal_pdf(
            batch['data_in'][:, 1:, :], batch['mask_in'][:, 1:, :], results["gen_values"][:, :-1, :], self.args)
        results["loss"] = -rec_likelihood.mean()

        results["mse"] = mean_squared_error(
            batch['data_in'][:, 1:, :], results["gen_values"][:, :-1, :], mask=batch['mask_in'][:, 1:, :]).detach()

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
