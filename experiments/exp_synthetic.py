import wandb

from experiments import BaseExperiment


class Exp_Synthetic(BaseExperiment):

    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_mse={results['mse']:.5f}")
        self.logger.info(f"val_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"val_mse": results['mse'], "epoch_id": epoch})
            wandb.log(
                {"val_forward_time": results['forward_time'], "epoch_id": epoch})
            wandb.log({"kldiv_z0": results["kldiv_z0"], "epoch_id": epoch})
            # temporally added
            wandb.log({"loss_ll_z": results["loss_ll_z"], "epoch_id": epoch})
            wandb.log(
                {"lat_variance": results["lat_variance"], "epoch_id": epoch})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_mse={results['mse']:.5f}")
        self.logger.info(f"test_mse_extrap={results['mse_extrap']:.5f}")
        self.logger.info(f"test_forward_time={results['forward_time']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log({"test_mse": results['mse'], "run_id": 1})
            wandb.log({"test_mse_extrap": results['mse_extrap'], "run_id": 1})
            wandb.log(
                {"test_forward_time": results['forward_time'], "run_id": 1})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']

    def get_model(self):
        print('current_ehr_variable_num: ' + str(self.variable_num))

        if self.args.ml_task == 'extrap':
            model = self.mf.initialize_extrap_model()
        elif self.args.ml_task == 'interp':
            model = self.mf.initialize_interp_model()
        elif self.args.ml_task == 'biclass':
            model = self.mf.initialize_biclass_model()
        elif self.args.ml_task == 'syn_extrap':
            model = self.mf.initialize_extrap_model()
        else:
            raise ValueError("Unknown")

        if self.args.model_type != "initialize":
            model = self.mf.reconstruct_models(
                model, self.proj_path/('temp/model/'+self.args.pre_model))

        return model
