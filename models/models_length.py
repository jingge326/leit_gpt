import torch
import torch.nn as nn
from models.ivp_vae import IVPVAE

from models.base_models.ivpvae_components import MLP_REGRE


class Leit_Length(IVPVAE):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 embedding_nn,
                 encoder_z0,
                 reconst_mapper,
                 diffeq_solver,
                 reg_input='zn',
                 train_w_reconstr=True):
        super().__init__(
            input_dim,
            latent_dim,
            embedding_nn,
            encoder_z0,
            reconst_mapper,
            diffeq_solver)

        self.train_w_reconstr = train_w_reconstr
        self.reg_input = reg_input
        self.regre_model = MLP_REGRE(self.latent_dim)

    def compute_prediction_results(self, batch, n_traj_samples=1, kl_coef=1.0):

        batch['truth'] = batch['truth'].repeat(n_traj_samples, 1, 1)

        results, forward_info = self.forward(batch)

        if self.reg_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.reg_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        reg_values = self.regre_model(x_input.squeeze(-2))

        # Calculating MSE and then mean it over all sample within a batch is
        # equivalent to this operation which calculate MSE on all values directly

        mse_reg = nn.MSELoss()(
            reg_values.reshape(-1, reg_values.shape[-1]),
            batch['truth'].reshape(-1, batch['truth'].shape[-1]))

        results["mse_reg"] = mse_reg.detach()

        mae_reg = nn.L1Loss()(
            reg_values.reshape(-1, reg_values.shape[-1]),
            batch['truth'].reshape(-1, batch['truth'].shape[-1]))
        results['mae_reg'] = mae_reg.detach()

        # reg1 = reg_values.reshape(
        #     reg_values.shape[0] * reg_values.shape[1], reg_values.shape[2])
        # reg2 = batch['truth'].reshape(
        #     batch['truth'].shape[0] * batch['truth'].shape[1], batch['truth'].shape[2])
        # mse_reg = nn.MSELoss()(reg1, reg2)
        # results["mse_reg"] = mse_reg.detach()

        results["label_predictions"] = reg_values.detach()

        loss = results['loss']

        if self.train_w_reconstr:
            loss = loss + 100 * mse_reg
        else:
            loss = mse_reg

        results["loss"] = torch.mean(loss)

        return results
