import time
import geotorch
import torch
import numpy as np
import torch.nn as nn
from typing import Iterable, Tuple, List, Union
from utils import TimeDistributed


def var_activation(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation function to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.log(torch.exp(x) + 1.0)


def var_activation_inverse(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(np.exp(x) - 1.0)


def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]


def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


def dadat(a: torch.Tensor, diag_mat: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    return bmv(a.square(), diag_mat)


def dadbt(a: torch.Tensor, diag_mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param b: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    return bmv(a * b, diag_mat)


class RKNCell(nn.Module):

    def __init__(self, latent_obs_dim: int, args, dtype: torch.dtype = torch.float32):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param args: args object, for configuring the cell
        :param dtype: dtype for input data
        """
        super(RKNCell, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod
        self.args = args

        self._dtype = dtype

        self._build_transition_model()

    # new code component
    def _var_activation(self):
        if self.args.trans_var_activation == 'exp':
            return torch.exp(self._log_transition_noise)
        elif self.args.trans_var_activation == 'relu':
            return torch.maximum(self._log_transition_noise, torch.zeros_like(self._log_transition_noise))
        elif self.args.trans_var_activation == 'square':
            return torch.square(self._log_transition_noise)
        elif self.args.trans_var_activation == 'abs':
            return torch.abs(self._log_transition_noise)
        else:
            return torch.log(torch.exp(self._log_transition_noise) + 1.0)

    # new code component
    def _var_activation_inverse(self):
        if self.args.trans_var_activation == 'exp':
            return np.log(self.args.trans_covar)
        elif self.args.trans_var_activation == 'relu':
            return self.args.trans_covar
        elif self.args.trans_var_activation == 'square':
            return np.sqrt(self.args.trans_covar)
        elif self.args.trans_var_activation == 'abs':
            return self.args.trans_covar
        else:
            return np.log(np.exp(self.args.trans_covar) - 1.0)

    def _compute_band_util(self, lod: int, bandwidth: int):
        self._num_entries = lod + 2 * np.sum(np.arange(lod - bandwidth, lod))
        np_mask = np.ones([lod, lod], dtype=np.float32)
        np_mask = np.triu(np_mask, -bandwidth) * np.tril(np_mask, bandwidth)
        mask = torch.tensor(np_mask, dtype=torch.bool)
        idx = torch.where(mask == 1)
        diag_idx = torch.where(idx[0] == idx[1])

        self.register_buffer("_idx0", idx[0], persistent=False)
        self.register_buffer("_idx1", idx[1], persistent=False)
        self.register_buffer("_diag_idx", diag_idx[0], persistent=False)

    def _unflatten_tm(self, tm_flat: torch.Tensor) -> torch.Tensor:
        tm = torch.zeros(
            tm_flat.shape[0], self._lod, self._lod, device=tm_flat.device, dtype=self._dtype)
        tm[:, self._idx0, self._idx1] = tm_flat
        return tm

    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor = None, delta_t: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """Forward pass trough the cell. 

        :param prior_mean: prior mean at time t
        :param prior_cov: prior covariance at time t
        :param obs: observation at time t
        :param obs_var: observation variance at time t
        :param obs_valid: boolean indicating whether observation at time t is valid
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :return: posterior mean at time t, posterior covariance at time t
                 prior mean at time t', prior covariance at time t', Kalman gain at time t
        """

        post_mean, post_cov, kalman_gain = self._update(
            prior_mean, prior_cov, obs, obs_var, obs_valid)

        next_prior_mean, next_prior_covar = self._predict(
            post_mean, post_cov, delta_t=delta_t)

        return post_mean, post_cov, next_prior_mean, next_prior_covar, kalman_gain

    def _build_coefficient_net(self, num_hidden: Iterable[int], activation: str) -> torch.nn.Sequential:
        """Builds the network computing the coefficients from the posterior mean. Currently only fully connected
        neural networks with same activation across all hidden layers supported
        :param num_hidden: number of hidden units per layer
        :param activation: hidden activation
        :return: coefficient network
        """
        layers = []
        prev_dim = self._lsd + 1 if self.args.t_sensitive_trans_net else self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self.args.num_basis))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers).to(dtype=self._dtype)

    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """

        # initialize eigenvectors E and eigenvalues d of state transition matrix
        if self.args.f_cru:
            self.E = nn.Linear(self._lsd, self._lsd, bias=False).double()
            self.d = nn.Parameter(
                1e-5 + torch.zeros(self.args.num_basis, self._lsd, dtype=torch.float32))

            if self.args.orthogonal:
                geotorch.orthogonal(self.E, 'weight')
                self.E.weight = torch.eye(
                    self._lsd, self._lsd, dtype=torch.float32)

        else:
            # build state independent basis matrices
            self._compute_band_util(
                lod=self._lod, bandwidth=self.args.bandwidth)
            self._tm_11_basis = nn.Parameter(torch.zeros(
                self.args.num_basis, self._num_entries, dtype=self._dtype))

            tm_12_init = torch.zeros(
                self.args.num_basis, self._num_entries, dtype=self._dtype)
            if self.args.rkn:
                tm_12_init[:, self._diag_idx] += 0.2 * torch.ones(self._lod)
            self._tm_12_basis = nn.Parameter(tm_12_init)

            tm_21_init = torch.zeros(
                self.args.num_basis, self._num_entries, dtype=self._dtype)
            if self.args.rkn:
                tm_21_init[:, self._diag_idx] -= 0.2 * torch.ones(self._lod)
            self._tm_21_basis = nn.Parameter(tm_21_init)

            self._tm_22_basis = nn.Parameter(torch.zeros(
                self.args.num_basis, self._num_entries, dtype=self._dtype))

            self._transition_matrices_raw = [
                self._tm_11_basis, self._tm_12_basis, self._tm_21_basis, self._tm_22_basis]

        self._coefficient_net = self._build_coefficient_net(self.args.trans_net_hidden_units,
                                                            self.args.trans_net_hidden_activation)

        init_log_trans_cov = self._var_activation_inverse()
        self._log_transition_noise = \
            nn.Parameter(nn.init.constant_(torch.empty(
                1, self._lsd, dtype=self._dtype), init_log_trans_cov))

    def get_transition_model(self, post_mean: torch.Tensor, delta_t: torch.Tensor = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :return: transition matrices for CRU and RKN or their eigenvectors and eigenvalues for f-CRU.
        """
        trans_net_input = torch.cat(
            [post_mean, delta_t[:, None]], 1) if self.args.t_sensitive_trans_net else post_mean
        coefficients = torch.reshape(self._coefficient_net(
            trans_net_input), [-1, self.args.num_basis, 1])  # [batchsize, c.num_basis, 1]

        if self.args.f_cru:
            # coefficients (batchsize, K, 1)
            eigenvalues = (coefficients * self.d).sum(dim=1)
            eigenvectors = self.E.weight
            transition = [eigenvalues, eigenvectors]

        else:
            # [batchsize, 93]
            tm11_flat = (coefficients * self._tm_11_basis).sum(dim=1)
            tm12_flat = (coefficients * self._tm_12_basis).sum(dim=1)
            tm21_flat = (coefficients * self._tm_21_basis).sum(dim=1)
            tm22_flat = (coefficients * self._tm_22_basis).sum(dim=1)

            # impose diagonal structure for transition matrix of RKN
            if self.args.rkn:
                tm11_flat[:, self._diag_idx] += 1.0
                tm22_flat[:, self._diag_idx] += 1.0

            transition = [self._unflatten_tm(
                x) for x in [tm11_flat, tm12_flat, tm21_flat, tm22_flat]]

        trans_cov = self._var_activation()

        return transition, trans_cov

    def _update(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs_mean: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs update step
        :param prior_mean: current prior state mean
        :param prior_cov: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current covariance mean
        :param obs_valid: flag if current time point is valid
        :return: current posterior state and covariance
        """
        cov_u, cov_l, cov_s = prior_cov

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = cov_u + obs_var
        q_upper = cov_u / denominator
        q_lower = cov_s / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + \
            torch.cat([q_upper * residual, q_lower * residual], -1)

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * cov_u
        new_covar_lower = cov_l - q_lower * cov_s
        new_covar_side = covar_factor * cov_s

        # ensures update only happens if an observation is given, otherwise posterior is set to prior
        obs_valid = obs_valid[..., None]
        masked_mean = new_mean.where(obs_valid, prior_mean)
        masked_covar_upper = new_covar_upper.where(obs_valid, cov_u)
        masked_covar_lower = new_covar_lower.where(obs_valid, cov_l)
        masked_covar_side = new_covar_side.where(obs_valid, cov_s)

        return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side], [q_upper, q_lower]

    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], delta_t: torch.Tensor = None) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step for regular time intervals (RKN variant)
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :param delta_t: ignored for discrete RKN
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        [tm11, tm12, tm21, tm22], trans_covar = self.get_transition_model(
            post_mean, delta_t)

        # prepare transition noise
        trans_covar_upper = trans_covar[..., :self._lod]
        trans_covar_lower = trans_covar[..., self._lod:]

        # predict next prior mean
        mu = post_mean[:, :self._lod]
        ml = post_mean[:, self._lod:]

        nmu = bmv(tm11, mu) + bmv(tm12, ml)
        nml = bmv(tm21, mu) + bmv(tm22, ml)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cu, cl, cs = post_covar
        ncu = dadat(tm11, cu) + 2.0 * dadbt(tm11, cs, tm12) + \
            dadat(tm12, cl) + trans_covar_upper
        ncl = dadat(tm21, cu) + 2.0 * dadbt(tm21, cs, tm22) + \
            dadat(tm22, cl) + trans_covar_lower
        ncs = dadbt(tm21, cu, tm11) + dadbt(tm22, cs, tm11) + \
            dadbt(tm21, cs, tm12) + dadbt(tm22, cl, tm12)
        return torch.cat([nmu, nml], dim=-1), [ncu, ncl, ncs]


# Continuous Discrete Kalman Cell: the prediction function assumes continuous states
# new code component
class CRUCell(RKNCell):
    def __init__(self, latent_obs_dim: int, args, dtype: torch.dtype = torch.float32):
        super(CRUCell, self).__init__(latent_obs_dim, args, dtype)

    def get_prior_covar_vanloan(self, post_covar, delta_t, Q, A, exp_A):
        """Computes Prior covariance matrix based on matrix fraction decomposition proposed by Van Loan.
        See Appendix A.2.1 in paper. This function is used for CRU.

        :param post_covar: posterior covariance at time t
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :param Q: diffusion matrix of Brownian motion in SDE that governs state evolution
        :param A: transition matrix
        :param exp_A: matrix exponential of (A * delta_t)
        :return: prior covariance at time t'
        """

        h2 = Q
        h3 = torch.zeros_like(h2)
        h4 = (-1) * torch.transpose(A, -2, -1)
        assert h2.shape == h3.shape == h4.shape, 'shapes must be equal (batchsize, latent_state_dim, latent_state_dim)'

        # construct matrix B (eq 27 in paper)
        # (batchsize, 2*latent_state_dim, 2*latent_state_dim)
        B = torch.cat((torch.cat((A, h2), -1), torch.cat((h3, h4), -1)), -2)
        exp_B = torch.matrix_exp(B * delta_t)
        # (batchsize, latent_state_dim, latent_state_dim)
        M1 = exp_B[:, :self._lsd, :self._lsd]
        M2 = exp_B[:, :self._lsd, self._lsd:]

        if torch.all(torch.isclose(M1, exp_A, atol=1e-8)) == False:
            print('---- ASSERTION M1 and exp_A are not identical ----')

        # compute prior covar (eq 28 in paper)
        C = torch.matmul(exp_A, post_covar) + M2
        prior_covar = torch.matmul(C, torch.transpose(exp_A, -2, -1))

        # for tensorboard plotting
        self.exp_B = exp_B
        self.M2 = M2

        return prior_covar

    def get_prior_covar_rome(self, post_covar, delta_t, Q, d, eigenvectors):
        """Computes prior covariance matrix based on the eigendecomposition of the transition matrix proposed by
        Rome (1969) https://ieeexplore.ieee.org/document/1099271. This function is used for f-CRU.

        :param post_covar: posterior covariance at time t
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :param Q: diffusion matrix of Brownian motion in SDE that governs state evolution
        :param d: eigenvalues of transition matrix
        :param eigenvectors: eigenvectors of transition matrix
        :return: prior covariance at time t'
        """

        jitter = 0  # 1e-8
        if self.args.orthogonal:
            eigenvectors_inverse = torch.transpose(eigenvectors, -2, -1)
            eigenvectors_inverse_trans = eigenvectors
        else:
            # not used in paper (no speed up, unstable)
            eigenvectors_inverse = torch.inverse(eigenvectors)
            eigenvectors_inverse_trans = torch.transpose(
                torch.inverse(eigenvectors), -1, -2)

        # compute Sigma_w of current time step (eq 22 in paper)
        Sigma_w = torch.matmul(eigenvectors_inverse, torch.matmul(
            post_covar, eigenvectors_inverse_trans))

        # compute D_tilde with broadcasting (eq 23 in paper)
        D_tilde = d[:, :, None] + d[:, None, :]
        exp_D_tilde = torch.exp(D_tilde * delta_t)

        # compute S (eq 24 in paper)
        S = torch.matmul(eigenvectors_inverse, torch.matmul(
            Q, eigenvectors_inverse_trans))

        # compute Sigma_w of next time step with elementwise multiplication/division (eq 25 in paper)
        Sigma_w_next = (S * exp_D_tilde - S) / \
            (D_tilde + jitter) + Sigma_w * exp_D_tilde

        # compute prior_covar (eq 26 in paper)
        prior_covar = torch.matmul(eigenvectors, torch.matmul(
            Sigma_w_next, torch.transpose(eigenvectors, -1, -2)))

        return prior_covar

    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], delta_t: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs continuous prediction step for irregularly sampled data 
        :param post_mean: last posterior mean
        :param post_cov: last posterior covariance
        :param delta_t: time delta to next observation
        :return: next prior state mean and covariance
        """

        delta_t = delta_t[:, None, None] if delta_t is not None else 1
        transition, trans_covar = self.get_transition_model(post_mean, delta_t)
        trans_covar = torch.diag_embed(
            trans_covar.repeat(post_mean.shape[0], 1))

        # build full covariance matrix
        post_cu, post_cl, post_cs = [torch.diag_embed(x) for x in post_covar]
        post_covar = torch.cat(
            (torch.cat((post_cu, post_cs), -1), torch.cat((post_cs, post_cl), -1)), -2)

        if self.args.f_cru:
            eigenvalues, eigenvectors = transition

            # compute prior mean (eq 21 in paper)
            exp_D = torch.diag_embed(
                torch.exp(eigenvalues * delta_t.squeeze(-1)))
            eigenvectors_inverse = torch.transpose(
                eigenvectors, -2, -1) if self.args.orthogonal else torch.inverse(eigenvectors)
            exp_A = torch.matmul(eigenvectors, torch.matmul(
                exp_D, eigenvectors_inverse))
            prior_mean = bmv(exp_A, post_mean)

            # compute prior covariance (eq 22 - 26 in paper)
            prior_covar = self.get_prior_covar_rome(
                post_covar, delta_t, trans_covar, eigenvalues, eigenvectors)

        else:
            [tm11, tm12, tm21, tm22] = transition
            A = torch.cat((torch.cat((tm11, tm12), -1),
                          torch.cat((tm21, tm22), -1)), -2)
            exp_A = torch.matrix_exp(A * delta_t)

            prior_mean = bmv(exp_A, post_mean)
            prior_covar = self.get_prior_covar_vanloan(
                post_covar, delta_t, trans_covar, A, exp_A)
            self.A = A

        # extract diagonal elements of prior covariance
        ncu = torch.diagonal(
            prior_covar[:, :self._lod, :self._lod], dim1=-1, dim2=-2)
        ncl = torch.diagonal(
            prior_covar[:, self._lod:, self._lod:], dim1=-1, dim2=-2)
        ncs = torch.diagonal(
            prior_covar[:, :self._lod, self._lod:], dim1=-1, dim2=-2)
        ncs2 = torch.diagonal(
            prior_covar[:, self._lod:, :self._lod], dim1=-1, dim2=-2)

        if torch.all(torch.isclose(ncs, ncs2, atol=1e-2)) == False:
            print('---- ASSERTION ncs not identical ----')

        # for tensorboard plotting
        self.exp_A = exp_A
        self.trans_covar = trans_covar

        return prior_mean, [ncu, ncl, ncs]


class CRULayer(nn.Module):

    def __init__(self, latent_obs_dim, args, dtype=torch.float32):
        super().__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._cell = RKNCell(latent_obs_dim, args, dtype) if args.rkn else CRUCell(
            latent_obs_dim, args, dtype)

    def forward(self, latent_obs, obs_vars, initial_mean, initial_cov, obs_valid=None, time_points=None):
        """Passes the entire observation sequence sequentially through the Kalman component

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating if observation is valid 
        :param time_points: timestamp of the observation
        """

        # prepare list for return
        prior_mean_list = []
        prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]
        kalman_gain_list = [[], []]

        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov
        T = latent_obs.shape[1]

        # iterate over sequence length
        for i in range(T):
            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
            delta_t = time_points[:, i+1] - time_points[:,
                                                        i] if time_points is not None and i < T-1 else torch.ones_like(latent_obs)[:, 0, 0]
            post_mean, post_cov, next_prior_mean, next_prior_cov, kalman_gain = \
                self._cell(prior_mean, prior_cov,
                           latent_obs[:, i], obs_vars[:, i], cur_obs_valid, delta_t=delta_t)
            #print(f'post_mean {post_mean.shape}, next_prior_mean {next_prior_mean.shape}')

            post_mean_list.append(post_mean)
            [post_cov_list[i].append(post_cov[i]) for i in range(3)]
            prior_mean_list.append(next_prior_mean)
            [prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]
            [kalman_gain_list[i].append(kalman_gain[i]) for i in range(2)]

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = [torch.stack(x, 1) for x in prior_cov_list]
        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]
        kalman_gains = [torch.stack(x, 1) for x in kalman_gain_list]

        return post_means, post_covs, prior_means, prior_covs, kalman_gains


class Encoder(nn.Module):

    def __init__(self, lod: int, enc_var_activation: str, output_normalization: str = "post"):
        """Gaussian Encoder, as described in RKN ICML Paper (if output_normalization=post)
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param enc_var_activation: activation function for latent observation noise
        :param output_normalization: when to normalize the output:
            - post: after output layer 
            - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
            - none: (or any other string) not at all

        """
        super(Encoder, self).__init__()
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                               "not found by the optimizer"
        self._mean_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        self.enc_var_activation = enc_var_activation
        self._output_normalization = output_normalization

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = obs
        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        log_var = self._log_var_layer(h)

        if self.enc_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.enc_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.enc_var_activation == 'square':
            var = torch.square(log_var)
        elif self.enc_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.enc_var_activation == 'elup1':
            var = torch.exp(log_var).where(log_var < 0.0, log_var + 1.0)
        else:
            raise Exception('Variance activation function unknown.')
        return mean, var


class SplitDiagGaussianDecoder(nn.Module):

    def __init__(self, lod: int, out_dim: int, dec_var_activation: str):
        """ Decoder for low dimensional outputs as described in the paper. This one is "split", i.e., there are
        completely separate networks mapping from latent mean to output mean and from latent cov to output var
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be a vector, images not supported by this decoder)
        :train_conf: configurate dict for training
        """
        self.dec_var_activation = dec_var_activation
        super(SplitDiagGaussianDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim

        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        assert isinstance(self._hidden_layers_mean, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"

        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()
        assert isinstance(self._hidden_layers_var, nn.ModuleList), "_build_hidden_layers_var needs to return a " \
                                                                   "torch.nn.ModuleList or else the hidden weights " \
                                                                   "are not found by the optimizer"

        self._out_layer_mean = nn.Linear(
            in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(
            in_features=num_last_hidden_var, out_features=out_dim)

    def _build_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor, latent_cov: Iterable[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward pass of decoder
        :param latent_mean:
        :param latent_cov:
        :return: output mean and variance
        """
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)

        if self.dec_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.dec_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.dec_var_activation == 'square':
            var = torch.square(log_var)
        elif self.dec_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.dec_var_activation == 'elup1':
            var = elup1(log_var)
        else:
            raise Exception('Variance activation function unknown.')
        return mean, var


class BernoulliDecoder(nn.Module):

    def __init__(self, lod: int, out_dim: int, args):
        """ Decoder for image output
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be images)
        :param args: parsed arguments
        """
        super(BernoulliDecoder, self).__init__()
        self._latent_obs_dim = lod
        self._out_dim = out_dim

        self._hidden_layers, num_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
            "torch.nn.ModuleList or else the hidden weights " \
            "are not found by the optimizer"
        self._out_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=num_last_hidden, out_channels=1, kernel_size=2, stride=2, padding=5),
                                        nn.Sigmoid())

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor) \
            -> torch.Tensor:
        """ forward pass of decoder
        :param latent_mean
        :return: output mean
        """
        h_mean = latent_mean
        for layer in self._hidden_layers:
            h_mean = layer(h_mean)
            #print(f'decoder: {h_mean.shape}')
        mean = self._out_layer(h_mean)
        #print(f'decoder mean {mean.shape}')
        return mean


class CRU(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.target_dim = args.variable_num
        self.input_dim = args.variable_num * 2
        self.hidden_units = args.cru_hidden_units
        lsd = args.latent_dim
        bernoulli_output = False

        self._device = torch.device(args.device)

        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2)
        else:
            raise Exception('Latent state dimension must be even number.')
        self.args = args
        self.time_start = 0

        # parameters
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = self.args.lr
        self.bernoulli_output = bernoulli_output
        # main model

        self._cru_layer = CRULayer(
            latent_obs_dim=self._lod, args=args).to(self._device)

        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(self._lod, output_normalization=self._enc_out_normalization,
                      enc_var_activation=args.enc_var_activation).to(dtype=torch.float32)

        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=self.target_dim, args=args).to(
                self._device, dtype=torch.float32), num_outputs=1, low_mem=True)
            self._enc = TimeDistributed(
                enc, num_outputs=2, low_mem=True).to(self._device)

        else:
            SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
            SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
            self._dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=self.target_dim, dec_var_activation=args.dec_var_activation).to(
                dtype=torch.float32), num_outputs=2).to(self._device)
            self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(
            self._device, dtype=torch.float32)
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float32))
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float32))
        self._ics = torch.zeros(1, self._lod).to(
            self._device, dtype=torch.float32)

    def _build_enc_hidden_layers(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))

        layers.append(nn.Linear(self.hidden_units, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))

        layers.append(nn.Linear(self.hidden_units, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))
        # size last hidden
        return nn.ModuleList(layers).to(dtype=torch.float32), self.hidden_units

    def _build_dec_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod,
                      out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units),

            nn.Linear(in_features=self.hidden_units,
                      out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units),

            nn.Linear(in_features=self.hidden_units,
                      out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units)
        ]).to(dtype=torch.float32), self.hidden_units

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod,
                      out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units)
        ]).to(dtype=torch.float32), self.hidden_units

    def forward(self, obs_batch: torch.Tensor, time_points: torch.Tensor = None, obs_valid: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass on a batch
        :param obs_batch: batch of observation sequences
        :param time_points: timestamps of observations
        :param obs_valid: boolean if timestamp contains valid observation
        """
        self.time_start = time.time()
        y, y_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(y, y_var, self._initial_mean,
                                                                                  [var_activation(self._log_icu), var_activation(
                                                                                      self._log_icl), self._ics],
                                                                                  obs_valid=obs_valid, time_points=time_points)
        # output an image
        if self.bernoulli_output:
            out_mean = self._dec(post_mean)
            out_var = None

        # output prediction for the next time step
        elif self.args.ml_task == 'one_step_ahead_prediction':
            out_mean, out_var = self._dec(
                prior_mean, torch.cat(prior_cov, dim=-1))

        # output filtered observation
        else:
            out_mean, out_var = self._dec(
                post_mean, torch.cat(post_cov, dim=-1))

        intermediates = {
            'post_mean': post_mean,
            'post_cov': post_cov,
            'prior_mean': prior_mean,
            'prior_cov': prior_cov,
            'kalman_gain': kalman_gain,
            'y': y,
            'y_var': y_var
        }

        return out_mean, out_var, intermediates
