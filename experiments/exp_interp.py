from experiments import BaseExperiment


class Exp_Interp(BaseExperiment):

    def validation_step(self):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={results['mse']:.5f}")
        self.logger.info(f"loss_vae={results['loss_vae']:.5f}")
        self.logger.info(f"ll_loss_z={results['ll_loss_z']:.5f}")
        self.logger.info(f"ll_loss_x={results['ll_loss_x']:.5f}")

        return results['loss']

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_mse={results['mse']:.5f}")
        self.logger.info(f"test_loss_vae={results['loss_vae']:.5f}")
        self.logger.info(f"test_ll_loss_z={results['ll_loss_z']:.5f}")
        self.logger.info(f"test_ll_loss_x={results['ll_loss_x']:.5f}")
        return results['loss']
