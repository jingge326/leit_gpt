from experiments import BaseExperiment


class Exp_Interp(BaseExperiment):

    def validation_step(self):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={results['val_mse']:.5f}")
        self.logger.info(f"val_loss={results['val_loss']:.5f}")
        return results['loss']

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_mse={results['test_mse']:.5f}")
        self.logger.info(f"test_loss={results['test_loss']:.5f}")
        return results['loss']
