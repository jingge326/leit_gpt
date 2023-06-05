from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sklearn as sk

DATA_DIR = Path(__file__).parents[1] / 'data'


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def get_device(tensor):
    device = torch.device('cpu')
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor(
        [0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def split_train_val_test(data, train_frac=0.6, val_frac=0.2):
    n_samples = len(data)
    data_train = data[:int(n_samples * train_frac)]
    data_val = data[int(n_samples * train_frac)
                        :int(n_samples * (train_frac + val_frac))]
    data_test = data[int(n_samples * (train_frac + val_frac)):]
    return data_train, data_val, data_test


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def reverse(tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_dict_template():
    return {'observed_data': None,
            'observed_tp': None,
            'data_to_predict': None,
            'tp_to_predict': None,
            'observed_mask': None,
            'mask_predicted_data': None,
            'labels': None
            }


def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception('Zero!')

    if torch.isnan(data_norm).any():
        raise Exception('nans!')

    return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception('Zero!')

    if torch.isnan(data_norm).any():
        raise Exception('nans!')

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def shift_outputs(outputs, first_datapoint=None):
    outputs = outputs[:, :, :-1, :]

    if first_datapoint is not None:
        n_traj, n_dims = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
        outputs = torch.cat((first_datapoint, outputs), 2)
    return outputs


def split_data_extrap(data_dict, dataset=''):
    device = get_device(data_dict['data'])

    n_observed_tp = data_dict['data'].size(1) // 2
    if dataset == 'hopper':
        n_observed_tp = data_dict['data'].size(1) // 3

    split_dict = {'observed_data': data_dict['data'][:, :n_observed_tp, :].clone(),
                  'observed_tp': data_dict['time_steps'][:n_observed_tp].clone(),
                  'data_to_predict': data_dict['data'][:, n_observed_tp:, :].clone(),
                  'tp_to_predict': data_dict['time_steps'][n_observed_tp:].clone()}

    split_dict['observed_mask'] = None
    split_dict['mask_predicted_data'] = None
    split_dict['labels'] = None

    if ('mask' in data_dict) and (data_dict['mask'] is not None):
        split_dict['observed_mask'] = data_dict['mask'][:,
                                                        :n_observed_tp].clone()
        split_dict['mask_predicted_data'] = data_dict['mask'][:,
                                                              n_observed_tp:].clone()

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict['mode'] = 'extrap'
    return split_dict


def split_data_interp(data_dict):
    device = get_device(data_dict['data'])

    split_dict = {'observed_data': data_dict['data'].clone(),
                  'observed_tp': data_dict['time_steps'].clone(),
                  'data_to_predict': data_dict['data'].clone(),
                  'tp_to_predict': data_dict['time_steps'].clone()}

    split_dict['observed_mask'] = None
    split_dict['mask_predicted_data'] = None
    split_dict['labels'] = None

    if 'mask' in data_dict and data_dict['mask'] is not None:
        split_dict['observed_mask'] = data_dict['mask'].clone()
        split_dict['mask_predicted_data'] = data_dict['mask'].clone()

    if ('labels' in data_dict) and (data_dict['labels'] is not None):
        split_dict['labels'] = data_dict['labels'].clone()

    split_dict['mode'] = 'interp'
    return split_dict


def add_mask(data_dict):
    data = data_dict['observed_data']
    mask = data_dict['observed_mask']

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict['observed_mask'] = mask
    return data_dict


def split_and_subsample_batch(data_dict, args, data_type='train'):
    if data_type == 'train':
        # Training set
        if args.extrap == True:
            processed_dict = split_data_extrap(data_dict, dataset=args.data)
        else:
            processed_dict = split_data_interp(data_dict)
    else:
        # Test set
        if args.extrap == True:
            processed_dict = split_data_extrap(data_dict, dataset=args.data)
        else:
            processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)
    return processed_dict


def check_mask(data, mask):
    # check that 'mask' argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert((n_zeros + n_ones) == np.prod(list(mask.size())))

    # all masked out elements should be zeros
    assert(torch.sum(data[mask == 0.] != 0.) == 0)
