import torch
import pytorch_lightning as pl


class PL_Wrapper(pl.LightningModule):
    def __init__(self, core_model, args):
        super().__init__()
        self.args = args
        self.core_model = core_model

    def training_step(self, batch):
        results, _ = self.core_model(batch)
        self.log('train_loss', results['loss'], sync_dist=True)
        return results['loss']

    def validation_step(self, batch, batch_idx):
        results, _ = self.core_model.run_validation(batch)
        self.log('val_loss', results['loss'], on_step=True, on_epoch=True)
        self.log('val_mse', results['mse'], on_step=True, on_epoch=True)
        return results['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
