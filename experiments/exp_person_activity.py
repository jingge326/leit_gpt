from sklearn import model_selection
import torch
from torch.utils.data import DataLoader

from experiments.data_person_activity import PersonActivity, collate_fn_activity
from utils import PROJ_FOLDER, compute_results_all_batches, get_data_min_max, split_train_val_test
from models.model_factory import ModelFactory

from experiments import BaseExperiment
import utils


DATA_DIR = PROJ_FOLDER / 'data'


class Exp_Person_Activity(BaseExperiment):
    def get_model(self):
        mf = ModelFactory(args)
        print('current_ehr_variable_num: ' + str(self.variable_num))
        model = mf.initialize_multiclass_model()

        if args.model_type == 'reconstruct':
            print('former_ehr_variable_num: ' +
                  str(args.former_ehr_variable_num))
            if args.para_file_type == 'pl_ckpt':
                model = mf.reconstruct_pl_biclass_model(
                    model, args.load_para_path)
            else:
                model = mf.reconstruct_models(
                    model, args.load_para_path)

        elif args.model_type == 'load':
            if args.para_file_type == 'pl_ckpt':
                model = mf.load_pl_mortality_model(model, args.load_para_path)
            else:
                model.load_state_dict(torch.load(
                    args.load_para_path, map_location=args.device))

        return model

    def get_data(self):

        dataset_obj = PersonActivity(
            DATA_DIR, download=True, n_samples=args.num_samples, device=args.device)

        train_data, vali_test_data = model_selection.train_test_split(
            dataset_obj, train_size=0.8, shuffle=False)
        val_data, test_data = model_selection.train_test_split(
            vali_test_data, train_size=0.5, shuffle=False)

        input_dim = train_data[0][2].shape[-1]

        batch_size = min(
            min(len(dataset_obj), args.batch_size), args.num_samples)
        dltrain = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_fn_activity(batch, args, args.device, data_type='train'))
        dlval = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda batch: collate_fn_activity(batch, args, args.device, data_type='test'))
        dltest = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda batch: collate_fn_activity(batch, args, args.device, data_type='test'))

        return input_dim, dltrain, dlval, dltest

    def training_step(self, batch):
        results = self.model.compute_prediction_results(batch)
        return results['loss']

    def _get_loss(self, dl):
        loss = compute_results_all_batches(self.model, dl)
        return results['loss'], results['mse'], results["auroc"], results['ce_loss']

    def validation_step(self):
        loss, mse, acc, ce_loss = self._get_loss(self.dlval)
        self.logger.info(f'val_mse={mse:.5f}')
        self.logger.info(f'val_auroc={acc:.5f}')
        self.logger.info(f'val_ce_loss={ce_loss:.5f}')
        return loss

    def test_step(self):
        loss, mse, acc, ce_loss = self._get_loss(self.dltest)
        self.logger.info(f'test_mse={mse:.5f}')
        self.logger.info(f'test_auroc={acc:.5f}')
        self.logger.info(f'val_ce_loss={ce_loss:.5f}')
        return loss
