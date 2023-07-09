from experiments import BaseExperiment


class Exp_Interp(BaseExperiment):

    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={results['mse']:.5f}")
        self.logger.info(f"val_loss={results['loss']:.5f}")
        return results['loss']

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_mse={results['mse']:.5f}")
        self.logger.info(f"test_loss={results['loss']:.5f}")
        return results['loss']
