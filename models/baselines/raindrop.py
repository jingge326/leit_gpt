import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from models.base_models.ivpvae_components import MLP_1L_ReLU

from models.baselines.ob_propagation import Observation_progation


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(
            0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pos_encoding = torch.cat([torch.sin(scaled_time), torch.cos(
            scaled_time)], axis=-1)  # T x B x d_model
        pos_encoding = pos_encoding.type(torch.FloatTensor)

        return pos_encoding

    def forward(self, P_time):
        pos_encoding = self.getPE(P_time)
        return pos_encoding


class Raindrop(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        variable_num = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_start = 0
        self.model_type = 'Transformer'
        self.d_model = self.args.d_ob * self.args.variable_num

        self.global_structure = torch.ones(
            self.args.variable_num, self.args.variable_num)

        self.encoder = nn.Linear(
            self.args.variable_num*self.args.d_ob, self.args.variable_num*self.args.d_ob)

        self.pos_encoder = PositionalEncodingTF(
            self.args.dim_pos_encoder, self.args.num_times, self.args.MAX)

        if self.args.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(
                self.args.variable_num*(self.args.d_ob+self.args.dim_pos_encoder), self.args.nhead, self.args.nhid, self.args.dropout)
        else:
            encoder_layers = TransformerEncoderLayer(
                self.d_model+self.args.dim_pos_encoder, self.args.nhead, self.args.nhid, self.args.dropout)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.args.nlayers)

        self.adj = torch.ones(
            [self.args.variable_num, self.args.variable_num]).to(self.args.device)

        self.dilator_relu = MLP_1L_ReLU(
            self.args.variable_num*self.args.d_ob, self.args.variable_num*self.args.d_ob)

        self.ob_propagation = Observation_progation(in_channels=self.args.num_times*self.args.d_ob,
                                                    out_channels=self.args.num_times*self.args.d_ob,
                                                    heads=1,
                                                    n_nodes=self.args.variable_num,
                                                    ob_dim=self.args.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=self.args.num_times*self.args.d_ob,
                                                           out_channels=self.args.num_times*self.args.d_ob,
                                                           heads=1,
                                                           n_nodes=self.args.variable_num,
                                                           ob_dim=self.args.d_ob)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.args.n_classes),
        )

        self.aggreg = self.args.aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.args.dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def compute(self, batch):
        """Input to the model:
        data = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        self.time_start = time.time()
        times = batch['times_in']
        data = batch['data_in']
        missing_mask = batch['mask_in']
        lengths = batch['lengths']

        batch_size = data.shape[1]
        data = torch.repeat_interleave(data, self.args.d_ob, dim=-1)
        h = self.dilator_relu(data)
        pos_encoding = self.pos_encoder(times).to(self.args.device)
        h = self.dropout(h)

        mask = torch.arange(self.args.num_times)[
            None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(self.args.device)

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.to(self.args.device)
            adj[torch.eye(self.args.variable_num).bool()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = data.shape[1]
            n_step = data.shape[0]
            output = torch.zeros(
                [n_step, batch_size, self.args.variable_num*self.args.d_ob]).to(self.args.device)

            use_beta = False
            if use_beta == True:
                alpha_all = torch.zeros(
                    [int(edge_index.shape[1]/2), batch_size]).to(self.args.device)
            else:
                alpha_all = torch.zeros(
                    [edge_index.shape[1],  batch_size]).to(self.args.device)
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pos_encoding[:, unit, :]

                stepdata = stepdata.reshape(
                    [n_step, self.args.variable_num, self.args.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(
                    self.args.variable_num, n_step*self.args.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                                                        use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view(
                    [self.args.variable_num, n_step, self.args.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape(
                    [-1, self.args.variable_num*self.args.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.args.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size,
                                        self.args.variable_num, self.args.d_ob)
            extended_pe = pos_encoding.unsqueeze(2).repeat(
                [1, 1, self.args.variable_num, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size,
                                 self.args.variable_num*(self.args.d_ob+self.args.dim_pos_encoder))
        else:
            output = torch.cat([output, pos_encoding], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.args.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros(
                    [batch_size, self.args.variable_num, self.args.d_ob+self.args.dim_pos_encoder]).to(self.args.device)
                extended_missing_mask = missing_mask.view(
                    -1, batch_size, self.args.variable_num)
                for se in range(self.args.variable_num):
                    r_out = r_out.view(-1, batch_size,
                                       self.args.variable_num, (self.args.d_ob+self.args.dim_pos_encoder))
                    out = r_out[:, :, se, :]
                    len = torch.sum(
                        extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(
                        out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view(
                    [-1, self.args.variable_num*(self.args.d_ob+self.args.dim_pos_encoder)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        forward_info = {"local_structure_regularization": distance.detach()}

        return output, forward_info
