from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from experiments.data_mimic import PretrainDataModule
from models.model_factory import ModelFactory


class Exp_M4_Pretrain():
    def __init__(self, args: Namespace):
        self.args = args
        self.proj_path = Path(__file__).parents[1]
        if self.args.leit_model == 'classic_rnn':
            self.args.exp_name = '_'.join(
                [args.leit_model, args.ml_task, args.data, args.rnn_cell, args.model_type, args.embedding_method, args.freeze_opt, str(args.train_w_reconstr), args.test_info])
        elif self.args.leit_model == 'ivp_vae':
            self.args.exp_name = '_'.join(
                [args.leit_model, args.ml_task, args.data, args.ivp_solver, args.model_type, args.embedding_method, args.freeze_opt, str(args.train_w_reconstr), 'm'+str(args.train_w_mask), args.test_info])
        elif self.args.leit_model == 'mtan':
            self.args.exp_name = '_'.join(
                [args.leit_model, args.ml_task, args.data, args.model_type, args.freeze_opt, str(args.train_w_reconstr), args.test_info])
        elif self.args.leit_model == 'mtan_ivp':
            self.args.exp_name = '_'.join(
                [args.leit_model, args.ml_task, args.data, args.model_type, args.freeze_opt, str(args.train_w_reconstr), args.test_info])
        elif self.args.leit_model == 'ckconv':
            self.args.exp_name = '_'.join(
                [args.leit_model, args.ml_task, args.data, args.model_type, args.freeze_opt, str(args.train_w_reconstr), args.test_info])
        else:
            raise ValueError('Leit model unknown!')

    def run(self) -> None:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.proj_path / 'temp/pl_checkpoint',
            filename=self.args.exp_name + "_{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, mode="min")

        data_module = PretrainDataModule(self.args, self.proj_path)
        mf = ModelFactory(self.args)
        model = mf.initialize_pretrain_model()

        if self.args.model_type == 'reconstruct':
            print('former_ehr_variable_num: ' +
                  str(self.args.former_ehr_variable_num))
            if self.args.para_file_type == 'pl_ckpt':
                model = mf.reconstruct_pl_biclass_model(
                    model, self.args.load_para_path)
            else:
                model = mf.reconstruct_biclass_model(
                    model, self.args.load_para_path)

        elif self.args.model_type == 'load':
            if self.args.para_file_type == 'pl_ckpt':
                model = mf.load_pl_pretrain_model(self.args.load_para_path)

            else:
                model.load_state_dict(torch.load(
                    self.args.load_para_path, map_location=self.args.device))

        logger = TensorBoardLogger(
            save_dir=self.proj_path / 'log/lightning_tb_log', version=1, name=self.args.exp_name)

        if self.args.dev_mode == 'debug':
            trainer = pl.Trainer(
                accelerator='cpu', logger=logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                callbacks=[checkpoint_callback, early_stop_callback])
        elif self.args.dev_mode == 'run':
            trainer = pl.Trainer(
                gpus=self.args.gpus, accelerator='gpu',
                strategy='ddp', logger=logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                callbacks=[checkpoint_callback, early_stop_callback])
        elif self.args.dev_mode == 'resume':
            trainer = pl.Trainer(
                gpus=self.args.gpus, accelerator='gpu',
                strategy='ddp', logger=logger,
                min_epochs=self.args.epochs_min,
                max_epochs=self.args.epochs_max,
                callbacks=[checkpoint_callback, early_stop_callback],
                resume_from_checkpoint=self.args.ckpt_path)
        else:
            raise ValueError(
                f'Invalid dev_mode: {self.args.dev_mode}')

        trainer.fit(model, data_module)

    def finish():
        pass
