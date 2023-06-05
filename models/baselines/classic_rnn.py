import time
import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRUCell

import utils


class ClassicRNN(nn.Module):
    def __init__(
            self,
            args,
            concat_mask=True,
            n_units=100):

        super(ClassicRNN, self).__init__()
        self.args = args
        self.time_start = 0
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.register_buffer('zero_tensor', torch.zeros([1]))
        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))

        variable_num = int(self.args.variable_num)
        if concat_mask:
            variable_num = variable_num * 2

        self.rnn_cell = GRUCell(
            variable_num + 1, self.rnn_hidden_dim)  # +1 for delta t

        self.rec_lyr = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, self.args.variable_num))

        utils.init_network_weights(self.rec_lyr)

    def forward(self, batch):
        self.time_start = time.time()

        times = batch['times_in']
        values = batch['data_in']
        masks = batch['mask_in']
        t_exist = times.gt(torch.zeros_like(times))

        self.zero_delta_t = self.zero_tensor.repeat(values.size(0), 1)
        delta_ts = times[:, 1:] - times[:, :-1]
        delta_ts = torch.cat((delta_ts, self.zero_delta_t), dim=1)

        hidden = torch.zeros(
            (values.size(0), self.rnn_hidden_dim), device=self.args.device)
        input_series = torch.cat((values, masks), -1)

        t_exist_h = t_exist.unsqueeze(-1).repeat(1, 1, self.rnn_hidden_dim)
        for i in range(input_series.size(1)):
            delta_t = delta_ts[:, i:i+1]
            input = input_series[:, i, :]
            input_w_t = torch.cat((input, delta_t), -1).squeeze(1)
            t_e = t_exist_h[:, i, :]

            prev_hidden = hidden
            hidden = self.rnn_cell(input_w_t, hidden)
            hidden = torch.where(t_e, hidden, prev_hidden)

        forward_info = {'hidden_state': hidden}
        return forward_info

    def generate_seq(self, first_hidden, delta_ts, cell, mask=None, decoder=None):
        first_input = decoder(first_hidden)
        n_steps = mask.size(1)
        outputs = []
        hidden = first_hidden
        rnn_input = first_input
        for i in range(n_steps):
            delta_t = delta_ts[:, i:i+1]
            if mask is not None:
                mask_i = mask[:, i, :]
                rnn_input = torch.cat((rnn_input, mask_i), -1)

            input_w_t = torch.cat((rnn_input, delta_t), -1).squeeze(1)
            hidden = cell(input_w_t, hidden)
            rnn_output = decoder(hidden)
            outputs.append(rnn_output)
            rnn_input = rnn_output

        outputs = torch.stack(outputs, 0).permute(1, 0, 2).unsqueeze(0)
        return outputs

    def run_validation(self, batch):
        return self.forward(batch)
