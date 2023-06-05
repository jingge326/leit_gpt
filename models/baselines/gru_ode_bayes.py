import time
import torch
import math
import numpy as np
import torch.nn as nn

from itertools import compress

from models.ivp_solvers import CouplingFlow, ODEModel, ResNetFlow, GRUFlow


class NNFOwithBayesianJumps(nn.Module):
    # NNFOwithBayesianJumps: Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self, args):
        super().__init__()
        input_size = args.variable_num
        p_hidden = args.gob_p_hidden
        prep_hidden = args.gob_prep_hidden
        mixing = args.mixing
        self.args = args
        self.time_start = 0

        self.p_model = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, p_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(p_hidden, 2 * input_size, bias=True),
        )

        if args.ivp_solver == 'ode':
            if args.odenet == 'gru':
                gru = FullGRUODECell_Autonomous(args.rnn_hidden_dim, bias=True)
                self.odeint = ODEModel(args.rnn_hidden_dim, gru, None, None,
                                       None, self.args.ode_solver, args.solver_step, args.atol, args.rtol)
            else:
                self.odeint = ODEModel(args.rnn_hidden_dim, args.odenet, [args.rnn_hidden_dim] * args.hidden_layers, args.activation,
                                       args.final_activation, self.args.ode_solver, args.solver_step, args.atol, args.rtol)

        else:
            if args.ivp_solver == 'couplingflow':
                flow = CouplingFlow
            elif args.ivp_solver == 'resnetflow':
                flow = ResNetFlow
            elif args.ivp_solver == 'gruflow':
                flow = GRUFlow
            else:
                raise NotImplementedError
            self.odeint = flow(args.rnn_hidden_dim, args.flow_layers, [args.rnn_hidden_dim] * args.hidden_layers,
                               args.time_net, args.time_rnn_hidden_dim, invertible=bool(args.invertible))

        self.input_size = input_size
        self.mixing = mixing
        self.gru_obs = GRUObservationCellLogvar(
            input_size, args.rnn_hidden_dim, prep_hidden, bias=True)

        self.apply(init_weights)

    def forward(self, batch, return_path=False):

        times = batch["times_in"]
        num_obs = batch["lengths_in"]
        X = batch["data_in"]
        M = batch["mask_in"]
        val_times = batch["times_out"]
        num_obs_out = batch["lengths_out"]

        self.time_start = time.time()

        h = torch.zeros(X.shape[0], self.args.rnn_hidden_dim).to(X)
        p = self.p_model(h)
        num_obs = num_obs.to(h.device)
        loss_1, loss_2 = 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            stepwise_mask = num_obs > ind
            current_times = torch.Tensor(
                [x[ind] for x in times[stepwise_mask.cpu()]])
            diff = current_times - last_times[stepwise_mask]

            solution = self.odeint(h[stepwise_mask].unsqueeze(
                1), diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[stepwise_mask] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[stepwise_mask])

            # zero_tens = torch.Tensor([0]).to(h.device)
            # ii = torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1]
            X_slice = X[stepwise_mask, ind, :]
            M_slice = M[stepwise_mask, ind, :]

            h, losses = self.gru_obs(h, p, X_slice, M_slice, stepwise_mask)
            real_NLL.append(losses.mean())

            assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[stepwise_mask])

            t_loss_start = time.time()
            loss_2 = loss_2 + \
                compute_KL_loss(p_obs=p, X_obs=X_slice,
                                M_obs=M_slice, logvar=True)
            t_loss_end = time.time()
            self.time_start += t_loss_end - t_loss_start

            last_times[stepwise_mask] = current_times

        if val_times is not None:
            for ind in range(0, int(torch.max(num_obs_out).item())):
                stepwise_mask = num_obs_out > ind
                current_times = torch.Tensor(
                    [x[ind] for x in list(compress(val_times, stepwise_mask.cpu()))])
                diff = current_times - last_times[stepwise_mask]

                solution = self.odeint(h[stepwise_mask].unsqueeze(
                    1), diff.view(-1, 1, 1).to(h.device))

                temp = h.clone()
                temp[stepwise_mask] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[stepwise_mask])

                if return_path:
                    path_p.append(p)

                last_times[stepwise_mask] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2


class FullGRUODECell_Autonomous(nn.Module):

    def __init__(self, hidden_size, bias=True):
        super().__init__()

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, inp):
        h, diff = inp[0], inp[1]
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        # change of variables
        dh = ((1 - z) * (u - h)) * diff
        return dh, torch.zeros_like(diff)


class GRUObservationCellLogvar(nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d = nn.GRUCell(
            prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(
            std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(
            0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p_obs, X_obs, M_obs, i_obs):
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        log_lik_c = np.log(np.sqrt(2*np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        assert not losses.sum() != losses.sum()

        gru_input = torch.stack(
            [X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(
            gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()

        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (gru_input * M_obs).permute(1, 2,
                                                0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp = self.gru_d(gru_input, h[i_obs].clone())
        h[i_obs] = temp.clone()

        return h, losses


def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.pow(torch.abs(var) + 1e-5, 0.5)

    return (gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std, sigma_2=obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return (torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1, 2)+torch.pow((mu_1 - mu_2), 2)) / (2*sigma_2**2) - 0.5)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)
