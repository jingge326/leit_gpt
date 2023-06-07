import time
import torch
import torch.nn as nn
from models.gpts import GPTS

from experiments.utils_metrics import compute_binary_CE_loss


class GPTS_PreTrain(GPTS):
    def __init__(self, args):
        super().__init__(args)
        # Classification
        self.args = args
        self.attn_inte_lyr = nn.Sequential(
            nn.Linear(args.attn_dim, 1),
            nn.ReLU())
        self.dropout = nn.Dropout(args.dropout)

    def compute_prediction_results(self, batch):
        forward_info = self.forward(batch)
        score = self.attn_inte_lyr(forward_info['latent_states'])
        score = self.softmax_with_mask(score, batch['exist_times'], dim=1)
        score = self.dropout(score)
        c_input = torch.sum(score * forward_info['latent_states'], dim=-2)

        # squeeze to remove the time dimension
        label_pred = self.classifier(c_input)
        results = {}
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred.squeeze(), batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["val_loss"] = results["ce_loss"]
        results["label_predictions"] = label_pred.detach()

        loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)

