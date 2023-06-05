###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn

import utils

from torch.nn.modules.rnn import GRUCell, RNNCellBase
from torch.nn.parameter import Parameter

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Exponential decay of the hidden states for RNN
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/

# Exp decay between hidden states


class GRUCellExpDecay(RNNCellBase):
    def __init__(self, input_size, input_size_for_decay, hidden_size, bias=True):
        super(GRUCellExpDecay, self).__init__(
            input_size, hidden_size, bias, num_chunks=3)

        self.register_buffer('zero', torch.zeros([1]))
        self.register_buffer('one', torch.ones([1]))

        self.input_size_for_decay = input_size_for_decay
        self.decay = nn.Sequential(nn.Linear(input_size_for_decay, 1),)
        utils.init_network_weights(self.decay)

    def gru_exp_decay_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
        # INPORTANT: assumes that cum delta t is the last dimension of the input
        batch_size, n_dims = input.size()

        # "input" contains the data, mask and also cumulative deltas for all inputs
        cum_delta_ts = input[:, -self.input_size_for_decay:]
        data = input[:, :-self.input_size_for_decay]

        decay_zero = self.zero
        decay_one = self.one
        decay = torch.exp(- torch.min(torch.max(decay_zero,
                          self.decay(cum_delta_ts)), decay_one * 1000))

        hidden = hidden * decay

        gi = torch.mm(data, w_ih.t()) + b_ih
        gh = torch.mm(hidden, w_hh.t()) + b_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, input, hx=None):
        if hx is None:
            hx = self.zero.repeat(input.size(0), self.hidden_size)
        #self.check_forward_hidden(input, hx, '')

        return self.gru_exp_decay_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Imputation with a weighed average of previous value and empirical mean
# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
def get_cum_delta_ts(delta_ts, mask):
    n_traj, n_tp, n_dims = mask.size()

    cum_delta_ts = delta_ts.unsqueeze(-1).repeat(1, 1, n_dims)
    missing_index = np.where(mask.cpu().numpy() == 0)

    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx]
        j = missing_index[1][idx]
        k = missing_index[2][idx]

        if j != 0 and j != (n_tp-1):
            cum_delta_ts[i, j+1, k] = cum_delta_ts[i,
                                                   j+1, k] + cum_delta_ts[i, j, k]

    cum_delta_ts = cum_delta_ts / cum_delta_ts.max()  # normalize

    return cum_delta_ts


# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
# very slow
def impute_using_input_decay(data, delta_ts, mask, w_input_decay, b_input_decay):
    n_traj, n_tp, n_dims = data.size()

    cum_delta_ts = delta_ts.unsqueeze(-1).repeat(1, 1, n_dims)
    missing_index = np.where(mask.cpu().numpy() == 0)

    data_last_obsv = np.copy(data.cpu().numpy())
    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx]
        j = missing_index[1][idx]
        k = missing_index[2][idx]

        if j != 0 and j != (n_tp-1):
            cum_delta_ts[i, j+1, k] = cum_delta_ts[i,
                                                   j+1, k] + cum_delta_ts[i, j, k]
        if j != 0:
            data_last_obsv[i, j, k] = data_last_obsv[i,
                                                     j-1, k]  # last observation
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max()  # normalize

    data_last_obsv = torch.Tensor(data_last_obsv).to(utils.get_device(data))

    zeros = torch.zeros([n_traj, n_tp, n_dims]).to(utils.get_device(data))
    decay = torch.exp(- torch.min(torch.max(zeros,
                                            w_input_decay * cum_delta_ts + b_input_decay), zeros + 1000))

    data_means = torch.mean(data, 1).unsqueeze(1)

    data_imputed = data * mask + \
        (1-mask) * (decay * data_last_obsv + (1-decay) * data_means)
    return data_imputed


def locf_impute_cum_dt(values, masks, times, t_exist):
    cum_delta_ts = torch.zeros_like(values)
    values_locf = torch.zeros_like(values)
    former_value = torch.zeros_like(values[:, 0, :])
    times_all = times.unsqueeze(-1).repeat(1, 1, values.size(-1))
    t_exist = t_exist.unsqueeze(-1).repeat(1, 1, values.size(-1))
    former_time = times_all[:, 0, :]
    for t in range(values.size(1)):
        t_mask = masks[:, t, :]
        t_value = values[:, t, :]
        former_value = t_mask * t_value + (1 - t_mask) * former_value
        values_locf[:, t, :] = former_value * t_exist[:, t, :]
        cum_delta_ts[:, t, :] = (
            times_all[:, t, :] - former_time) * t_exist[:, t, :]
        former_time = t_mask * times_all[:, t, :] + (1 - t_mask) * former_time

    return values_locf, cum_delta_ts


# adapted from GRU-D implementation: https://github.com/zhiyongc/GRU-D/
# very slow
def impute_decay_input_ts(data, delta_ts, mask, w_input_decay, b_input_decay):
    n_traj, n_tp, n_dims = data.size()

    cum_delta_ts = delta_ts.unsqueeze(-1).repeat(1, 1, n_dims)
    missing_index = np.where(mask.cpu().numpy() == 0)

    data_last_obsv = np.copy(data.cpu().numpy())
    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx]
        j = missing_index[1][idx]
        k = missing_index[2][idx]

        if j != 0 and j != (n_tp-1):
            cum_delta_ts[i, j+1, k] = cum_delta_ts[i,
                                                   j+1, k] + cum_delta_ts[i, j, k]
        if j != 0:
            data_last_obsv[i, j, k] = data_last_obsv[i,
                                                     j-1, k]  # last observation
    cum_delta_ts = cum_delta_ts / cum_delta_ts.max()  # normalize

    data_last_obsv = torch.Tensor(data_last_obsv).to(utils.get_device(data))

    zeros = torch.zeros([n_traj, n_tp, n_dims]).to(utils.get_device(data))
    decay = torch.exp(- torch.min(torch.max(zeros,
                                            w_input_decay * cum_delta_ts + b_input_decay), zeros + 1000))

    data_means = torch.mean(data, 1).unsqueeze(1)

    data_imputed = data * mask + \
        (1-mask) * (decay * data_last_obsv + (1-decay) * data_means)
    return data_imputed, cum_delta_ts

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def run_rnn(inputs, delta_ts, cell, first_hidden=None,
            mask=None, feed_previous=False, n_steps=0,
            decoder=None, input_decay_params=None,
            feed_previous_w_prob=0.,
            masked_update=True):
    if (feed_previous or feed_previous_w_prob) and decoder is None:
        raise Exception(
            "feed_previous is set to True-- please specify RNN decoder")

    if n_steps == 0:
        n_steps = inputs.size(1)

    if (feed_previous or feed_previous_w_prob) and mask is None:
        mask = torch.ones(
            (inputs.size(0), n_steps, inputs.size(-1))).to(utils.get_device(inputs))

    if isinstance(cell, GRUCellExpDecay):
        if input_decay_params is not None:
            w_input_decay, b_input_decay = input_decay_params
            inputs, cum_delta_ts = impute_decay_input_ts(inputs, delta_ts, mask,
                                                         w_input_decay, b_input_decay)
        else:
            cum_delta_ts = get_cum_delta_ts(delta_ts, mask)

    else:
        if input_decay_params is not None:
            w_input_decay, b_input_decay = input_decay_params
            inputs = impute_using_input_decay(inputs, delta_ts, mask,
                                              w_input_decay, b_input_decay)

    all_hiddens = []
    hidden = first_hidden

    if hidden is not None:
        all_hiddens.append(hidden)
        n_steps -= 1

    for i in range(n_steps):
        delta_t = delta_ts[:, i:i+1]
        if i == 0:
            rnn_input = inputs[:, i]
        elif feed_previous:
            rnn_input = decoder(hidden)
        elif feed_previous_w_prob > 0:
            feed_prev = np.random.uniform() > feed_previous_w_prob
            if feed_prev:
                rnn_input = decoder(hidden)
            else:
                rnn_input = inputs[:, i]
        else:
            rnn_input = inputs[:, i]

        if mask is not None:
            mask_i = mask[:, i, :]
            rnn_input = torch.cat((rnn_input, mask_i), -1)

        if isinstance(cell, GRUCellExpDecay):
            cum_delta_t = cum_delta_ts[:, i]
            input_w_t = torch.cat((rnn_input, cum_delta_t), -1).squeeze(1)
        else:
            input_w_t = torch.cat((rnn_input, delta_t), -1).squeeze(1)

        prev_hidden = hidden
        hidden = cell(input_w_t, hidden)

        if masked_update and (mask is not None) and (prev_hidden is not None):
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            summed_mask = (torch.sum(mask_i, -1, keepdim=True) > 0).float()
            assert(not torch.isnan(summed_mask).any())
            hidden = summed_mask * hidden + (1-summed_mask) * prev_hidden

        all_hiddens.append(hidden)

    all_hiddens = torch.stack(all_hiddens, 0)
    all_hiddens = all_hiddens.permute(1, 0, 2).unsqueeze(0)
    return hidden, all_hiddens


def generate_seq(first_hidden, delta_ts, cell, mask=None, decoder=None):
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
