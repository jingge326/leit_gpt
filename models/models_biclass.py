import time
import torch
import torch.nn as nn
from experiments.utils_mtan import compute_log_normal_pdf
from models.att_ivpvae import AttIVPVAE
from models.base_models.mtan_components import create_classifier
from models.baselines.classic_rnn import ClassicRNN
from models.baselines.grud import GRUD
from models.baselines.mtan import MTAN
from models.baselines.raindrop import Raindrop
from models.baselines.mtan_ivp import MTANIVP
from models.baselines.ckconv import CKCONV
from models.baselines.red_vae import REDVAE
from models.gpts import GPTS
from models.ivp_auto import IVPAuto

from models.ivp_vae import IVPVAE
from models.base_models.ivpvae_components import BinaryClassifier
from experiments.utils_metrics import compute_binary_CE_loss
from models.ivpvae_old import IVPVAE_OLD


class IVPVAE_BiClass(IVPVAE):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["val_loss"] = results["ce_loss"]
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class GPTS_BiClass(GPTS):
    def __init__(self, args):
        super().__init__(args)
        # Classification
        self.args = args
        if self.args.gpts_output == "all":
            self.attn_inte_lyr = nn.Sequential(
                nn.Linear(args.n_embd, 1),
                nn.ReLU())
            self.dropout = nn.Dropout(args.dropout)
        self.classifier = BinaryClassifier(self.args.n_embd)

    def softmax_with_mask(self, input, mask, dim):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
        output = torch.exp(input) * mask
        output = output / torch.sum(output, dim=dim, keepdim=True)
        return output

    def compute_prediction_results(self, batch):
        results = self.forward(batch)

        if self.args.gpts_output == "all":
            score = self.attn_inte_lyr(results["latent_states"])
            score = self.softmax_with_mask(score, batch['exist_times'], dim=1)
            score = self.dropout(score)
            c_input = torch.sum(score * results["latent_states"], dim=-2)
        else:
            exist_edge = torch.logical_xor(torch.concat([batch['exist_times'][:, 1:], torch.zeros(
                batch['exist_times'].shape[0], 1).to(batch["data_in"])], dim=-1), batch['exist_times'])
            c_input = (results["latent_states"] *
                       exist_edge.unsqueeze(-1)).sum(dim=-2)

        # squeeze to remove the time dimension
        label_pred = self.classifier(c_input)

        times = batch['times_in']
        delta_ts = times[:, 1:] - times[:, :-1]
        latent_states = results["latent_states"][:, :-1, :]
        B, T, C = latent_states.size()
        latent_states = latent_states.view(
            B, T, self.args.nhead, C // self.args.nhead).unsqueeze(-2)
        delta_ts = delta_ts.view(B, T, 1, 1).repeat(1, 1, self.args.nhead, 1)
        evolved_states = self.evolve(latent_states, delta_ts).view(B, T, C)
        value_pred = self.lm_head(evolved_states)

        results['forward_time'] = time.time() - self.time_start

        # Compute Negative log-likelihood loss
        rec_likelihood = compute_log_normal_pdf(
            batch['data_in'][:, 1:, :], batch['mask_in'][:, 1:, :], value_pred, self.args)
        nll_loss = -rec_likelihood.mean()

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred.squeeze(), batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["val_loss"] = results["ce_loss"]
        results["label_predictions"] = label_pred.detach()

        loss = nll_loss + ce_loss * self.args.ratio_ce
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class IVPVAE_OLD_BiClass(IVPVAE_OLD):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["val_loss"] = results["ce_loss"]
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class IVPAuto_BiClass(IVPAuto):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class AttIVPVAE_BiClass(AttIVPVAE):
    def __init__(
            self,
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class REDVAE_BiClass(REDVAE):
    def __init__(self, args, z0_diffeq_solver, diffeq_solver):

        super().__init__(args, z0_diffeq_solver, diffeq_solver)

        # Classification
        self.args = args
        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch, k_iwae=1):

        results, forward_info = self.forward(batch, k_iwae)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class ClassicRNN_BiClass(ClassicRNN):
    def __init__(self, args):

        super().__init__(args)
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch):
        forward_info = self.forward(batch)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['hidden_state']
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2)).unsqueeze(0)
        results = {}
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class MTAN_BiClass(MTAN):
    def __init__(self, args, encoder_z0, decoder):

        super().__init__(args, encoder_z0, decoder)

        self.classifier = create_classifier(
            latent_dim=self.args.latent_dim, nhidden=self.args.gen_hidden)
        self.criterion = nn.CrossEntropyLoss()

    def compute_prediction_results(self, batch, k_iwae=1):
        results, forward_info = self.forward(batch, k_iwae)
        x_input = forward_info['initial_state']

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input)
        results['forward_time'] = time.time() - self.time_start

        labels = batch['truth'].type(torch.long).repeat(k_iwae, 1, 1).view(-1)

        # Compute CE loss
        ce_loss = self.criterion(label_pred, labels)
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred[:,
                                                  1].reshape(k_iwae, -1).detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch, k_iwae=self.args.k_iwae)


class MTANIVP_BiClass(MTANIVP):
    def __init__(
            self,
            args,
            encoder_z0,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            encoder_z0,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = BinaryClassifier(self.args.latent_dim)

    def compute_prediction_results(self, batch):
        results, forward_info = self.forward(batch)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['initial_state']
        elif self.args.classifier_input == 'zn':
            x_input = forward_info['sol_z'][:, :, -1, :]
        else:
            raise NotImplementedError

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2))
        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class Raindrop_BiClass(Raindrop):
    def __init__(self, args):

        super().__init__(args)
        self.args = args

        self.classifier = BinaryClassifier(
            self.d_model + self.args.dim_pos_encoder)

    def compute_prediction_results(self, batch):
        output, _ = self.compute(batch)
        x_input = output.unsqueeze(0)
        results = {}

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input)
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()
        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class CKCONV_BiClass(CKCONV):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.finallyr = torch.nn.Linear(
            in_features=self.args.no_hidden, out_features=1)
        # Initialize finallyr
        self.finallyr.weight.data.normal_(mean=0.0, std=0.01)
        self.finallyr.bias.data.fill_(value=0.0)

    def compute_prediction_results(self, batch):
        data = batch['data']
        output = self.forward(data)
        # # modified by Jingge
        # output = self.compressor(output).squeeze(-1)
        # label_pred = self.finallyr(output).squeeze(-1)
        label_pred = self.finallyr(output[:, :, -1]).unsqueeze(0)
        results = {}
        results['forward_time'] = time.time() - self.time_start

        results = {}
        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()
        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class GRUD_BiClass(GRUD):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.classifier = BinaryClassifier(self.args.rnn_hidden_dim)

    def compute_prediction_results(self, batch):
        results = {}
        forward_info = self.forward(batch)
        x_input = forward_info['hidden_state']

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input.squeeze(-2)).unsqueeze(0)
        results['forward_time'] = time.time() - self.time_start

        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()
        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
