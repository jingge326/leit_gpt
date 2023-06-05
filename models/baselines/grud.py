"""
PyTorch GRU-D model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3
import time
import torch
import torch.nn as nn

from libs.pypots.imputation.brits import TemporalDecay
from models.base_models.locf import LOCF
from models.base_models.rnn_utils import locf_impute_cum_dt


class GRUD(nn.Module):
    def __init__(self, args, n_units=100):
        super().__init__()
        self.n_features = args.variable_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.device = args.device
        self.time_start = 0
        self.locf = LOCF()
        self.args = args

        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_dim, diag=False)

        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True)

        self.rec_lyr = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, self.args.variable_num))

    def forward(self, batch):
        self.time_start = time.time()

        times = batch['times_in']
        values = batch['data_in']
        masks = batch['mask_in']
        t_exist = times.gt(torch.zeros_like(times))

        empirical_mean = torch.sum(
            masks * values, dim=[0, 1]) / (torch.sum(masks, dim=[0, 1]) + 1e-10)
        values_imputed, cum_delta_ts = locf_impute_cum_dt(
            values, masks, times, t_exist)

        hidden_state = torch.zeros(
            (values.size(0), self.rnn_hidden_dim), device=self.device)

        t_exist_h = t_exist.unsqueeze(-1).repeat(1, 1, self.rnn_hidden_dim)
        for t in range(times.size(-1)):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            cdt = cum_delta_ts[:, t, :]  # delta, time gap
            x_im = values_imputed[:, t, :]
            t_e = t_exist_h[:, t, :]

            gamma_x = self.temp_decay_x(cdt)
            gamma_h = self.temp_decay_h(cdt)

            hidden_state = hidden_state * gamma_h

            x_h = gamma_x * x_im + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h

            inputs = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state_new = self.rnn_cell(inputs, hidden_state)
            hidden_state_new = torch.where(
                t_e, hidden_state_new, hidden_state)
            hidden_state = hidden_state_new

        forward_info = {'hidden_state': hidden_state}
        return forward_info

    def generate_seq(self, first_hidden, delta_ts):
        x = self.rec_lyr(first_hidden)
        m = torch.ones_like(x)
        delta_ts = delta_ts.unsqueeze(-1).repeat(1, 1, self.args.variable_num)
        outputs = []
        hidden_state = first_hidden
        for t in range(delta_ts.size(1)):
            delta_t = delta_ts[:, t, :]  # delta, time gap
            gamma_h = self.temp_decay_h(delta_t)
            hidden_state = hidden_state * gamma_h
            inputs = torch.cat([x, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(inputs, hidden_state)
            x = self.rec_lyr(hidden_state)
            outputs.append(x)

        outputs = torch.stack(outputs, 0).permute(1, 0, 2)
        return outputs

    def run_validation(self, batch):
        return self.forward(batch)
