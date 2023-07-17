import copy
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from experiments.data_mimic import M4PretrainDataModule
from models.model_factory import ModelFactory


class Exp_M4_Pretrain():
    def __init__(self, args: Namespace):
        self.args = args
        self.proj_path = Path(args.proj_path)
        self.tags = [self.args.train_obj,
                     self.args.ml_task,
                     self.args.model_type,
                     "nhead"+str(self.args.nhead),
                     "nlyrs"+str(self.args.mhatt_n_layer),
                     "bsize"+str(self.args.batch_size),
                     args.test_info]

        self.args.exp_name = '_'.join(
            self.tags + [("r"+str(args.random_state))])

        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    def run(self) -> None:
        os.environ["WANDB__SERVICE_WAIT"] = "1800"
        logger = WandbLogger(project="leit_gpt",
                             config=copy.deepcopy(
                                 dict(self.args._get_kwargs())),
                             group="_".join(self.tags),
                             tags=self.tags,
                             name="r"+str(self.args.random_state))

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.proj_path / 'temp/pl_checkpoint',
            filename=self.args.exp_name + "_{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=self.args.patience, mode="min")

        data_module = M4PretrainDataModule(self.args, self.proj_path, logger)
        mf = ModelFactory(self.args, logger=logger)
        model = mf.initialize_pretrain_model()

        if self.args.model_type == 'reconstruct':
            print('former_ehr_variable_num: ' +
                  str(self.args.former_ehr_variable_num))
            if self.args.para_file_type == 'pl_ckpt':
                model = mf.reconstruct_pl_biclass_model(
                    model, self.args.load_para_path)
            else:
                model = mf.reconstruct_models(
                    model, self.args.load_para_path)

        elif self.args.model_type == 'load':
            if self.args.para_file_type == 'pl_ckpt':
                model = mf.load_pl_pretrain_model(self.args.load_para_path)

            else:
                model.load_state_dict(torch.load(
                    self.args.load_para_path, map_location=self.args.device))

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
