import logging
import math
import random

import numpy as np
import torch
from kernels.tropical import (
    tropical_mmpp,
    tropical_maxp,
    tropical_minp,
    leaky_tropical_minp,
    leaky_tropical_maxp,
    tropical_multiminp,
    tropical_multimax,
    solidmaxmin,
    track_activation,
    tropical_mmpp_conv,
)
from weight_updates import max_update, min_update
from torch import nn


class SumTanh(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X):
        summed_out = torch.tanh(self.W.view(1, self.W.shape[0], self.W.shape[1]) + X.view(X.shape[0], 1, X.shape[1]))
        return torch.sum(summed_out, dim=-1)


class SolidLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, init_std=None, kernel=True):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_std:
            torch.nn.init.normal_(self.W, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W)

        self.kernel = kernel

    def forward(self, X):
        if self.kernel:
            return tropical_mmpp.apply(X, self.W)
        else:
            sum_var = self.W.view(1, self.W.shape[0], self.W.shape[1]) + X.view(X.shape[0], 1, X.shape[1])
            return sum_var.min(axis=-1).values + sum_var.max(axis=-1).values


class MinMaxSigmoid(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_std=None,
        is_tracking=False,
        kernel=True,
    ):
        super().__init__()
        self.W_MAX = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.W_MIN = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_std:
            torch.nn.init.normal_(self.W_MAX, std=init_std)
            torch.nn.init.normal_(self.W_MIN, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W_MAX)
            torch.nn.init.xavier_uniform_(self.W_MIN)
        self.W_activation_MAX = None
        self.W_activation_MIN = None
        if is_tracking:
            self.W_activation_MAX = torch.zeros_like(self.W_MAX)
            self.W_activation_MIN = torch.zeros_like(self.W_MIN)
        self.kernel = kernel

    def forward(self, X):
        return solidmaxmin.apply(X, self.W_MAX, self.W_MIN, self.W_activation_MAX, self.W_activation_MIN)

    def reset_weight_activation(self):
        self.W_activation_MIN = torch.zeros_like(self.W_MIN)
        self.W_activation_MAX = torch.zeros_like(self.W_MAX)

    def update_weights(self, lambda_, mode):
        if self.W_activation_MIN is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = min_update(self.W_MIN, self.W_activation_MIN, lambda_, mode)
        self.W_MIN.copy_(updated_W)

        if self.W_activation_MAX is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = min_update(self.W_MAX, self.W_activation_MAX, lambda_, mode)
        self.W_MAX.copy_(updated_W)


class ScaledMinPlus(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        is_tracking=False,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)
        self.factors = torch.nn.Parameter(torch.ones(out_features, 1))
        self.W_activation = None
        if is_tracking:
            self.W_activation = torch.zeros_like(self.W)

    def forward(self, X):
        sum_var = self.W.view(1, self.W.shape[0], self.W.shape[1]) + X.view(
            X.shape[0], 1, X.shape[1]
        ) * self.factors.view(1, self.W.shape[0], 1)
        minimum = sum_var.min(axis=-1)
        track_activation(self.W_activation, minimum.indices, "scaled_min")
        return minimum.values

    def get_factors(self):
        return self.factors

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        self.W_activation = torch.zeros_like(self.W)


class ScaledMaxPlus(torch.nn.Module):
    def __init__(self, in_features, out_features, is_tracking=False):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)
        self.factors = torch.nn.Parameter(torch.ones(out_features, 1))
        if is_tracking:
            self.W_activation = torch.zeros_like(self.W)

    def forward(self, X):
        sum_var = self.W.view(1, self.W.shape[0], self.W.shape[1]) + X.view(
            X.shape[0], 1, X.shape[1]
        ) * self.factors.view(1, self.W.shape[0], 1)
        maximum = sum_var.max(axis=-1)
        track_activation(self.W_activation, maximum.indices, "scaled_max")
        return maximum.values

    def get_factors(self):
        return self.factors

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        self.W_activation = torch.zeros_like(self.W)


class TropicalMinReLu(torch.nn.Module):
    def __init__(self, in_features, out_features, init_std=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X):
        x_relu = torch.clone(X)
        x_relu[x_relu < 0] = float("inf")
        return tropical_minp.apply(x_relu, self.W, self.W_activation, self.normalized_W)


class TropicalMaxReLu(torch.nn.Module):
    def __init__(self, in_features, out_features, init_std=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X):
        x_relu = torch.clone(X)
        x_relu[x_relu > 0] = float("-inf")
        return tropical_minp.apply(x_relu, self.W, self.W_activation, self.normalized_W)


class MinPlus(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_const=None,
        init_std=None,
        is_tracking=False,
        is_dropout=False,
        is_normalized=False,
        is_multi=False,
        dropout_probability_bias=None,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_const:
            torch.nn.init.zeros_(self.W)
        elif init_std:
            torch.nn.init.normal_(self.W, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W)
        self.is_dropout = is_dropout
        self.dropout_probability_bias = dropout_probability_bias
        self.dropout_probability = np.ones(in_features) / in_features
        self.W_activation = None
        if is_tracking or is_dropout:
            self.W_activation = torch.zeros_like(self.W)
        self.normalized_W = None
        if is_normalized:
            self.normalized_W = torch.zeros_like(self.W)
        self.is_multi = is_multi

    def forward(self, X):
        if self.is_multi:
            return tropical_multiminp.apply(X, self.W, self.W_activation)
        if self.is_dropout and self.training:
            idx = np.random.choice(range(self.W.shape[1]), size=1, p=self.dropout_probability)[0]
            dropout_W = torch.clone(self.W)

            if random.random() < self.dropout_probability_bias:
                dropout_W[:, idx] = float("inf")
            return tropical_minp.apply(X, dropout_W, self.W_activation, self.normalized_W)
        return tropical_minp.apply(X, self.W, self.W_activation, self.normalized_W)

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        if self.is_dropout and torch.sum(self.W_activation) > 0:
            self.dropout_probability = (
                (torch.sum(self.W_activation, dim=0) / torch.sum(self.W_activation)).cpu().detach().numpy()
            )
        self.W_activation = torch.zeros_like(self.W)

    def update_weights(self, lambda_, mode):
        if self.W_activation is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = min_update(self.W, self.W_activation, lambda_, mode)
        self.W.copy_(updated_W)

    def get_weight_regularizer(self):
        return -torch.sum(torch.abs(self.W))

    def get_normalized_weight_regularizer(self):
        if torch.sum(self.normalized_W) == 0:
            return torch.clone(self.W)
        return torch.sum(torch.abs(self.W - self.normalized_W)) / torch.numel(self.W)

    def reset_normalized_weight(self, lambda_):
        if self.normalized_W is None:
            logging.getLogger(__name__).warning("No normalized weight is tracked")
            return
        updated_W = (1 - lambda_) * self.W + lambda_ * self.normalized_W.to(self.W.device)
        self.W.copy_(updated_W)
        self.normalized_W = torch.zeros_like(self.W)


class Max_0_Plus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        sum_var = X.view(X.shape[0], 1, X.shape[1])
        return sum_var.max(axis=-1).values


class Max_B_Plus(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_const=None,
        init_std=None,
        is_tracking=False,
        is_dropout=False,
        is_normalized=False,
        is_multi=False,
        dropout_probability_bias=None,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_const:
            torch.nn.init.zeros_(self.W)
        elif init_std:
            torch.nn.init.normal_(self.W, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W)
        self.is_dropout = is_dropout
        self.dropout_probability_bias = dropout_probability_bias
        self.dropout_probability = np.ones(in_features) / in_features
        self.W_activation = None
        if is_tracking or is_dropout:
            self.W_activation = torch.zeros_like(self.W)
        self.normalized_W = None
        if is_normalized:
            self.normalized_W = torch.zeros_like(self.W)
        self.is_multi = is_multi

    def forward(self, X):
        if self.is_multi:
            return tropical_multimax.apply(X, self.W, self.W_activation)
        if self.is_dropout and self.training:
            idx = np.random.choice(range(self.W.shape[1]), size=1, p=self.dropout_probability)[0]
            dropout_W = torch.clone(self.W)
            if random.random() < self.dropout_probability_bias:
                dropout_W[:, idx] = -float("inf")
            return tropical_maxp.apply(X, dropout_W, self.W_activation, self.normalized_W)
        return tropical_maxp.apply(X, self.W, self.W_activation, self.normalized_W)

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        if self.is_dropout and torch.sum(self.W_activation) > 0:
            self.dropout_probability = (
                (torch.sum(self.W_activation, dim=0) / torch.sum(self.W_activation)).cpu().detach().numpy()
            )
        self.W_activation = torch.zeros_like(self.W)

    def update_weights(self, lambda_, mode):
        if self.W_activation is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = max_update(self.W, self.W_activation, lambda_, mode)
        self.W.copy_(updated_W)

    def get_weight_regularizer(self):
        return torch.sum(torch.abs(self.W))

    def get_normalized_weight_regularizer(self):
        if torch.sum(self.normalized_W) == 0:
            return torch.clone(self.W)
        return torch.sum(torch.abs(self.W - self.normalized_W)) / torch.numel(self.W)

    def reset_normalized_weight(self, lambda_):
        if self.normalized_W is None:
            logging.getLogger(__name__).warning("No normalized weight is tracked")
            return
        updated_W = (1 - lambda_) * self.W + lambda_ * self.normalized_W.to(self.W.device)
        self.W.copy_(updated_W)
        self.normalized_W = torch.zeros_like(self.W)


class LeakyMinPlus(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_std=None,
        is_random=False,
        leaky_factor=1,
        is_tracking=False,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_std:
            torch.nn.init.normal_(self.W, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W)
        self.W_activation = None
        if is_tracking:
            self.W_activation = torch.zeros_like(self.W)
        self.normalized_W = torch.zeros_like(self.W)
        self.leaky_factor = torch.nn.Parameter(torch.Tensor([leaky_factor]), False)
        self.is_random = torch.nn.Parameter(torch.Tensor([is_random]), False)

    def forward(self, X):
        return leaky_tropical_minp.apply(X, self.W, self.W_activation, self.leaky_factor, self.is_random)

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        self.W_activation = torch.zeros_like(self.W)

    def update_weights(self, lambda_, mode):
        if self.W_activation is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = min_update(self.W, self.W_activation, lambda_, mode)
        self.W.copy_(updated_W)

    def get_weight_regularizer(self):
        return -torch.sum(torch.abs(self.W))

    def get_normalized_weight_regularizer(self):
        if torch.sum(self.normalized_W) == 0:
            return torch.clone(self.W)
        return torch.sum(torch.abs(self.W - self.normalized_W)) / torch.numel(self.W)

    def reset_normalized_weight(self):
        if self.normalized_W is None:
            logging.getLogger(__name__).warning("No normalized weight is tracked")
            return
        self.normalized_W = torch.zeros_like(self.W)


class LeakyMax_B_Plus(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        init_std=None,
        is_random=False,
        leaky_factor=1,
        is_tracking=False,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        if init_std:
            torch.nn.init.normal_(self.W, std=init_std)
        else:
            torch.nn.init.xavier_uniform_(self.W)
        self.W_activation = None
        if is_tracking:
            self.W_activation = torch.zeros_like(self.W)
        self.normalized_W = None
        self.leaky_factor = torch.nn.Parameter(torch.Tensor([leaky_factor]), False)
        self.is_random = torch.nn.Parameter(torch.Tensor([is_random]), False)

    def forward(self, X):
        return leaky_tropical_maxp.apply(X, self.W, self.W_activation, self.leaky_factor, self.is_random)

    def get_weight_activation(self):
        return self.W_activation.cpu().detach().numpy()

    def reset_weight_activation(self):
        self.W_activation = torch.zeros_like(self.W)

    def update_weights(self, lambda_, mode):
        if self.W_activation is None:
            logging.getLogger(__name__).warning("No weight activation is tracked")
            return
        updated_W = max_update(self.W, self.W_activation, lambda_, mode)
        self.W.copy_(updated_W)

    def get_weight_regularizer(self):
        return torch.sum(torch.abs(self.W))

    def get_normalized_weight_regularizer(self):
        if torch.sum(self.normalized_W) == 0:
            return torch.clone(self.W)
        return torch.sum(torch.abs(self.W - self.normalized_W)) / torch.numel(self.W)

    def reset_normalized_weight(self):
        self.normalized_W = torch.zeros_like(self.W)


class SolidConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.stride = stride
        self.padding = padding

    def forward(self, X):
        return tropical_mmpp_conv(X.contiguous(), self.W, self.stride, self.padding)
