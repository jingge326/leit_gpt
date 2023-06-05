import torch
import torch.nn as nn
from models.base_models.mtan_components import create_classifier
from models.baselines.classic_rnn import ClassicRNN
from models.baselines.mtan import MTAN
from models.baselines.raindrop import Raindrop
from models.baselines.mtan_ivp import MTANIVP
from models.baselines.ckconv import CKCONV
from models.baselines.red_vae import REDVAE

from models.ivp_vae import IVPVAE
from models.base_models.ivpvae_components import BinaryClassifier, MultiClassifier
from experiments.utils_metrics import compute_binary_CE_loss


class IVPVAE_MultiClass(IVPVAE):
    def __init__(
            self,
            args,
            embedding_nn,
            encoder_z0,
            reconst_mapper,
            diffeq_solver):

        super().__init__(
            args,
            embedding_nn,
            encoder_z0,
            reconst_mapper,
            diffeq_solver)

        # Classification
        self.args = args

        self.classifier = MultiClassifier(self.args.latent_dim, self.args.num_class)
        self.criterion = nn.CrossEntropyLoss()

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
        # Compute CE loss
        labels = batch['truth'].repeat(self.args.k_iwae, *batch['truth']).view(-1)
        ce_loss = self.criterion(label_pred, labels)
        
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


class REDVAE_MultiClass(REDVAE):
    def __init__(self, args, z0_diffeq_solver, diffeq_solver):

        super().__init__(args, z0_diffeq_solver, diffeq_solver)

        # Classification
        self.args = args
        self.classifier = MultiClassifier(self.args.latent_dim, self.args.num_class)

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


class ClassicRNN_MultiClass(ClassicRNN):
    def __init__(self, args):

        super().__init__(args)
        self.args = args

        self.classifier = MultiClassifier(self.args.latent_dim, self.args.num_class)

    def compute_prediction_results(self, batch):
        results, forward_info = self.forward(batch)

        if self.args.classifier_input == 'z0':
            x_input = forward_info['hidden_state']
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


class MTAN_MultiClass(MTAN):
    def __init__(self, args, encoder_z0, decoder):

        super().__init__(args, encoder_z0, decoder)

        self.classifier = create_classifier(
            latent_dim=self.args.latent_dim, nhidden=self.args.gen_hidden)
        self.criterion = nn.CrossEntropyLoss()

    def compute_prediction_results(self, batch):
        results, forward_info = self.forward(batch)
        x_input = forward_info['initial_state']

        # squeeze to remove the time dimension
        label_pred = self.classifier(x_input)

        labels = batch['truth'].type(torch.long).repeat(self.args.k_iwae, *batch['truth']).view(-1)
        
    
        # Compute CE loss
        ce_loss = self.criterion(label_pred, labels)
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred[:,
                                                  1].reshape(self.args.k_iwae, -1).detach()

        loss = results['loss']
        if self.args.train_w_reconstr:
            loss = loss + ce_loss * self.args.ratio_ce
        else:
            loss = ce_loss
        results["loss"] = torch.mean(loss)

        return results


class MTANIVP_MultiClass(MTANIVP):
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

        self.classifier = MultiClassifier(self.args.latent_dim, self.args.num_class)

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


class Raindrop_MultiClass(Raindrop):
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
        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()
        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)


class CKCONV_MultiClass(CKCONV):
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
        # Compute CE loss
        ce_loss = compute_binary_CE_loss(label_pred, batch['truth'])
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["label_predictions"] = label_pred.detach()
        results["loss"] = torch.mean(ce_loss)

        return results

    def run_validation(self, batch):
        return self.compute_prediction_results(batch)
        