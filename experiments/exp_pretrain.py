import copy
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import sklearn
import torch
import wandb
from argparse import Namespace
from torch.utils.data import DataLoader

from experiments.data_mimic import MIMICDatasetGP, collate_fn_gpts, load_tvt
from models.model_factory import ModelFactory
from models.models_pretrain import GPTS_PreTrain
from utils import record_experiment


class Exp_Pretrain:

    def __init__(self, args: Namespace):
        self.args = args
        self.epochs_max = args.epochs_max
        self.patience = args.patience
        self.proj_path = Path(args.proj_path)
        self.mf = ModelFactory(self.args)
        self.tags = ["gpts",
                     "nhead"+str(self.args.nhead),
                     "nlyrs"+str(self.args.mhatt_n_layer),
                     "bsize"+str(self.args.batch_size),
                     args.test_info]

        self.args.exp_name = '_'.join(self.tags)

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)

        self._init_logger()
        self.device = torch.device(args.device)
        self.logger.info(f'Device: {self.device}')

        self.dltrain, self.dlval = self.get_data()

        self.model = GPTS_PreTrain(args=self.args).to(self.device)

        self.logger.info(f'num_params={self.model.num_params}')

        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = None
        if args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim, args.lr_scheduler_step, args.lr_decay)

    def _init_logger(self):

        logging.basicConfig(filename=self.proj_path / 'log' / (self.args.exp_name+'.log'),
                            filemode='w',
                            level=logging.INFO,
                            force=True)

        self.logger = logging.getLogger()

        if self.args.log_tool == 'wandb':
            # initialize weight and bias
            if self.args.model_type != "initialize":
                os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB__SERVICE_WAIT"] = "1800"
            wandb.init(
                project="leit_gpt",
                config=copy.deepcopy(dict(self.args._get_kwargs())),
                group="_".join(self.tags),
                tags=self.tags,
                name="r"+str(self.args.random_state))

    def get_data(self):

        m4_path = self.proj_path/"data/mimic4/processed/"
        data_train, data_validation, _ = load_tvt(
            self.args, m4_path, self.logger)

        train_dataset = MIMICDatasetGP(data_train, data_path=m4_path/"split")
        val_dataset = MIMICDatasetGP(
            data_validation, data_path=m4_path/"split")

        dl_train = DataLoader(
            dataset=train_dataset,
            collate_fn=lambda batch: collate_fn_gpts(
                batch, self.args.variable_num, self.args),
            shuffle=True,
            batch_size=self.args.batch_size)
        dl_val = DataLoader(
            dataset=val_dataset,
            collate_fn=lambda batch: collate_fn_gpts(
                batch, self.args.variable_num, self.args),
            shuffle=True,
            batch_size=self.args.batch_size)

        return dl_train, dl_val

    def update_model(self, model):
        if self.args.model_type == 'reconstruct':
            print('former_ehr_variable_num: ' +
                  str(self.args.former_ehr_variable_num))
            if self.args.para_file_type == 'pl_ckpt':
                model = self.mf.reconstruct_pl_biclass_model(
                    model, self.args.load_para_path)
            else:
                model = self.mf.reconstruct_models(
                    model, self.args.load_para_path)

        elif self.args.model_type == 'load':
            if self.args.para_file_type == 'pl_ckpt':
                model = self.mf.load_pl_mortality_model(
                    model, self.args.load_para_path)
            else:
                model.load_state_dict(torch.load(
                    self.args.load_para_path, map_location=self.args.device))

        if self.args.freeze_opt == 'odevae':
            model.embedding_nn.requires_grad_(False)
            model.encoder_z0.requires_grad_(False)
            model.diffeq_solver.requires_grad_(False)
            model.reconst_mapper.requires_grad_(False)
        elif self.args.freeze_opt == 'embedding':
            model.embedding_nn.requires_grad_(False)
        elif self.args.freeze_opt == 'embedding_nn_gc':
            model.embedding_nn.gc.requires_grad_(False)
        elif self.args.freeze_opt == 'flow':
            model.embedding_nn.requires_grad_(False)
            model.encoder_z0.z0_diffeq_solver.requires_grad_(False)
            model.diffeq_solver.requires_grad_(False)
        elif self.args.freeze_opt == 'encoder_flow':
            model.embedding_nn.requires_grad_(False)
            model.encoder_z0.z0_diffeq_solver.requires_grad_(False)
        elif self.args.freeze_opt == 'decoder':
            model.embedding_nn.requires_grad_(False)
            model.diffeq_solver.requires_grad_(False)

        return model

    def run(self) -> None:
        # Training loop parameters
        best_loss = float('inf')
        waiting = 0
        durations = []
        best_model = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.epochs_max):
            iteration = 1
            self.model.train()
            start_time = time.time()

            for batch in self.dltrain:
                # Single training step

                self.optim.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                if self.args.clip_gradient:
                    # Optional gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip)
                self.optim.step()

                self.logger.info(
                    f'[epoch={epoch:04d}|iter={iteration:04d}] train_loss={train_loss:.5f}')
                if self.args.log_tool == 'wandb':
                    wandb.log({"train_loss": train_loss})
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(
                f'[epoch={epoch:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            val_loss = self.validation_step(epoch)
            self.logger.info(f'[epoch={epoch:04d}] val_loss={val_loss:.5f}')
            if self.args.log_tool == 'wandb':
                wandb.log({"epoch_duration": epoch_duration, "epoch_id": epoch})
                wandb.log({"val_loss": val_loss, "epoch_id": epoch})

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                waiting = 0
            else:
                waiting += 1

            if waiting >= self.patience:
                break

            if self.args.log_tool == 'wandb':
                wandb.log(
                    {"lr": self.optim.param_groups[0]['lr'], "epoch_id": epoch})

        # Load best model
        self.model.load_state_dict(best_model)
        self.logger.info(f'epoch_duration_mean={np.mean(durations):.5f}')

        if self.args.log_tool == 'wandb':
            wandb.log({"epoch_duration_mean": np.mean(durations), "run_id": 1})

    def training_step(self, batch):
        return self.model(batch)["loss"]

    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_forward_time={results['forward_time']:.5f}")
        self.logger.info(f"val_mse={results['mse']:.5f}")
        if self.args.log_tool == "wandb":
            wandb.log(
                {"val_forward_time": results['forward_time'], "epoch_id": epoch})
            wandb.log({"val_mse": results['mse'], "epoch_id": epoch})
        if results['val_loss'] != 0:
            return results['val_loss']
        else:
            return results['loss']

    def compute_results_all_batches(self, dl):
        total = {}
        total['loss'] = 0
        total['likelihood'] = 0
        total['mse'] = 0
        total['mse_reg'] = 0
        total['mae_reg'] = 0
        total['mse_extrap'] = 0
        total['forward_time'] = 0
        total["val_loss"] = 0

        n_test_batches = 0

        for batch in dl:
            results = self.model.run_validation(batch)

            for key in total.keys():
                if results.get(key) is not None:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach()
                    total[key] += var

            n_test_batches += 1

        if n_test_batches > 0:
            for key, _ in total.items():
                total[key] = total[key] / n_test_batches

        return total

    def finish(self):
        record_experiment(self.args, self.model)
        torch.save(self.model.state_dict(), self.proj_path /
                   'temp/model' / (self.args.exp_name+'.pt'))
        logging.shutdown()
