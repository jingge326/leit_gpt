import torch
from torch.utils.data import DataLoader

from experiments.data_mimic import get_length_dataset
from utils import compute_results_all_batches
from models.model_factory import ModelFactory
from experiments.data_mimic import collate_fn_biclass

from experiments import BaseExperiment


class Exp_Length(BaseExperiment):

    def training_step(self, batch):
        results = self.model.compute_prediction_results(batch)
        return results['loss']

    def _get_loss(self, dl):
        loss = compute_results_all_batches(self.model, dl)
        return results['loss'], results['mse'], results['mse_reg'], results['mae_reg']

    def validation_step(self):
        loss, mse, mse_reg, mae_reg = self._get_loss(self.dlval)
        self.logger.info(f'val_mse={mse:.5f}')
        self.logger.info(f'val_mse_reg={mse_reg:.5f}')
        self.logger.info(f'val_mae_reg={mae_reg:.5f}')
        return loss

    def test_step(self):
        loss, mse, mse_reg, mae_reg = self._get_loss(self.dltest)
        self.logger.info(f'test_mse={mse:.5f}')
        self.logger.info(f'test_mse_reg={mse_reg:.5f}')
        self.logger.info(f'test_mae_reg={mae_reg:.5f}')
        return loss
