import numpy as np
import torch
import torch.nn as nn

from models.base_models.mtan_components import multiTimeAttention

import utils


def mask_of_timestamp(mask):
    mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
    assert (not torch.isnan(mask).any())
    return mask


class ExpLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return nn.functional.linear(input, self.weights.exp(), self.bias.exp())


class IVPVAE_Encoder_Linear(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver=None):
        super().__init__()

        self.lstm = nn.LSTMCell(input_dim, latent_dim)
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z2mu_mapper = Z_to_mu(self.latent_dim)
        self.z2std_mapper = Z_to_std(self.latent_dim)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.scale_nn = nn.Sequential(
            ExpLinear(1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent = self.run_odernn(data, mask, time_steps, run_backwards)
        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        return mean_z0, std_z0

    def run_odernn(self, data, mask, time_steps, run_backwards=True):
        assert run_backwards
        time_points_iter = range(0, time_steps.shape[1])
        time_points_iter = reversed(time_points_iter)
        # time_steps = torch.neg(time_steps)

        batch_size = data.size(0)
        # Before scaling, the timestamp was the time delta between the first chart time for each admission
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        mask_tt = mask_of_timestamp(mask)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)

            h = self.ivp_solver(h.unsqueeze(1), t).squeeze(1)

            assert (not torch.isnan(h).any())

            xi = data[:, i, :]
            scale = self.scale_nn(self.cos(h, xi).unsqueeze(-1))
            h_ = scale * h + (1 - scale) * xi

            assert (not torch.isnan(h).any())

            h = mask_tt[:, i, :] * h_ + (1 - mask_tt[:, i, :]) * h

            prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

        return h


class Softmax_with_mask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
        output = torch.exp(input) * mask
        output = output / torch.sum(output, dim=self.dim, keepdim=True)
        return output


class Z0_Attention_Net(nn.Module):

    def __init__(self, input_dim, attn_dropout=0.1):
        super().__init__()
        self.w = nn.Linear(input_dim, 1, bias=False)
        utils.init_network_weights(self.w)
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(attn_dropout)
        self.softmax_with_mask = Softmax_with_mask(dim=1)

    def forward(self, data, h, mask):
        attn = self.w(data)
        # attn = self.dropout(self.softmax(attn))
        attn = self.softmax_with_mask(attn, mask)
        output = torch.sum(attn * h, dim=1)
        return output


class IVPVAE_Encoder(nn.Module):
    def __init__(self, latent_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        if args.test_relu:
            self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        else:
            self.z2mu_mapper = Z_to_mu(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        latent = self.ivp_solver(data.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        latent = torch.sum(latent*mask_tt, dim=1) / mask_tt.sum(dim=1)

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        return mean_z0, std_z0


class IVPVAE_Encoder_logstd(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z0_mapper = Z0_Mapper(self.latent_dim)
        self.args = args

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        latent = self.ivp_solver(data.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        latent = torch.sum(latent*mask_tt, dim=1) / mask_tt.sum(dim=1)

        latent = self.z0_mapper(latent)
        mean_z0, logstd_z0 = latent.split(self.args.latent_dim, dim=-1)
        std_z0 = torch.exp(.5 * logstd_z0)

        return mean_z0, std_z0


class IVPVAE_Encoder_Inte(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        if args.test_relu:
            self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        else:
            self.z2mu_mapper = Z_to_mu(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        latent = self.ivp_solver(data.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        mean_z0 = torch.sum(mean_z0*mask_tt, dim=1) / mask_tt.sum(dim=1)
        std_z0 = torch.sum(std_z0*mask_tt, dim=1) / mask_tt.sum(dim=1)

        return mean_z0, std_z0


class IVPVAE_Encoder_Att(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z0_att_net = Z0_Attention_Net(latent_dim)
        if args.test_relu:
            self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        else:
            self.z2mu_mapper = Z_to_mu(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        latent = self.ivp_solver(data.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        latent = self.z0_att_net(latent, latent, mask_tt)
        # latent = self.z0_att_net(data, latent, mask_tt)

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        return mean_z0, std_z0


class IVPVAE_Encoder_Att_Rough(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z0_att_net = Z0_Attention_Net(args.variable_num*2)
        if args.test_relu:
            self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        else:
            self.z2mu_mapper = Z_to_mu(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

    def forward(self, data, mask, time_steps, rough_data, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        latent = self.ivp_solver(data.unsqueeze(-2),
                                 back_time_steps.unsqueeze(-1)).squeeze()

        latent = self.z0_att_net(rough_data, latent, mask_tt)
        # latent = self.z0_att_net(data, latent, mask_tt)

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        return mean_z0, std_z0


class IVPVAE_Encoder_MulAtt(nn.Module):
    def __init__(self, latent_dim, input_dim, ivp_solver, args):
        super().__init__()
        self.ivp_solver = ivp_solver
        self.latent_dim = latent_dim
        self.z0_att_net = Z0_Attention_Net(latent_dim)
        if args.test_relu:
            self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        else:
            self.z2mu_mapper = Z_to_mu(self.latent_dim)
            self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)

        self.embed_time = 64
        self.nhidden = 256
        self.register_buffer('query', torch.linspace(0, 1., 128))
        self.learn_emb = True
        self.att = multiTimeAttention(
            input_dim, self.nhidden, self.embed_time, 1)
        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time-1)
            self.linear = nn.Linear(1, 1)
        self.device = args.device

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, data, mask, time_steps, run_backwards=True, save_info=False):
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        # time_steps = torch.neg(time_steps)
        latent = self.travel_back_time(data, mask, time_steps)

        mean_z0 = self.z2mu_mapper(latent)
        std_z0 = self.z2std_mapper(latent)

        return mean_z0, std_z0

    def travel_back_time(self, data, mask, time_steps):

        # mask_dd = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))

        mask_tt = mask_of_timestamp(
            mask).repeat_interleave(data.shape[-1], dim=-1)
        back_time_steps = torch.neg(time_steps)

        h = self.ivp_solver(data.unsqueeze(-2),
                            back_time_steps.unsqueeze(-1)).squeeze()

        out = self.att(query, key, h, mask_tt)

        # h = self.z0_att_net(data, h, mask_tt)

        return out


class Embedding_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, truth, mask):
        x = torch.cat((truth, mask), -1)
        assert (not torch.isnan(x).any())
        out = self.layers(x)
        return out


class Embedding_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim)
        )
        utils.init_network_weights(self.layers, method='xavier_uniform_')

    def forward(self, truth, mask):
        x = torch.cat((truth, mask), -1)
        assert (not torch.isnan(x).any())
        return self.layers(x)


class Reconst_Mapper_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, data):
        truth = self.layers(data)
        return truth


class Reconst_Mapper_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
        utils.init_network_weights(self.layers, method='xavier_uniform_')

    def forward(self, data):
        truth = self.layers(data)
        return truth


class Reconst_DM_Mapper(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.Tanh(),
            nn.Linear(200, out_dim * 2)
        )
        utils.init_network_weights(self.layers)

    def forward(self, data):
        truth, mask = self.layers(data).chunk(2, dim=-1)
        return truth, mask


class Reconst_DM_Mapper_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim * 2)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, data):
        truth, mask = self.layers(data).chunk(2, dim=-1)
        return truth, mask


class Z_to_mu(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, latent_dim),)
        utils.init_network_weights(self.net)

    def forward(self, data):
        return self.net(data)


class Z_to_mu_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),)
        utils.init_network_weights(self.net, method='kaiming_uniform_')

    def forward(self, data):
        return self.net(data)


class Z0_Mapper(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim*2),)
        utils.init_network_weights(self.net, method='kaiming_uniform_')

    def forward(self, data):
        return self.net(data)


class Z_to_std_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.Softplus(),)
        utils.init_network_weights(self.net)

    def forward(self, data):
        return self.net(data)


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, x):
        return self.layers(x)


class MultiClassifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, num_class),
            nn.Softmax()
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, x):
        return self.layers(x)


class State2Distribution_Mapper(nn.Module):
    def __init__(self, latent_dim, z0_dim):
        super().__init__()

        self.s2d_mapper = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, z0_dim*2),)

        utils.init_network_weights(self.s2d_mapper)

    def forward(self, data):
        return self.s2d_mapper(data)


class MLP_REGRE(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self, z0_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z0_dim, z0_dim * 10),
            nn.ReLU(),
            nn.Linear(z0_dim * 10, z0_dim * 10),
            nn.ReLU(),
            nn.Linear(z0_dim * 10, 1)
        )
        utils.init_network_weights(self.layers)

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


class DilatorTanh(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(1, out_dim, bias=False)
        self.tanh = torch.nn.ReLU()

    def forward(self, obs):
        return self.tanh(self.linear(obs.unsqueeze(-1)))


class CompressorTanh(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 1, bias=False)
        self.tanh = torch.nn.ReLU()

    def forward(self, obs):
        return self.tanh(self.linear(obs).squeeze(-1))


class MLP_1L_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.relu = torch.nn.ReLU()
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, data_in):
        return self.relu(self.linear(data_in))


class Extrap_Fixed_Linear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        return out
