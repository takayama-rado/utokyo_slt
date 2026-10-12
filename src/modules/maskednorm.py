#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""maskednorm: Masked Normalization
-------------------------------------------------------------------------------

This script is arranged from
https://gist.github.com/amiasato/902fc14afa37a7537386f7b0c5537741
and
https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py

Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# Third party's modules
import torch
import torch.nn.functional as F
from torch import nn

# Local modules


# Execution settings
VERSION = "%(prog)s dev"


def insert_axes_to_mask(mask, target_size):
    """Insert the axis to mask to match target size.
    """
    _mask = mask
    _target = [t for i, t in enumerate(target_size) if i != 1]
    # If mask's temporary size is different from that of target,
    # we down sample mask.
    if _mask.size(1) != _target[1]:
        _mask = _mask.unsqueeze(1)
        dtype = _mask.dtype
        # Interpolate does not accept boolean type.
        _mask = F.interpolate(_mask.float(), size=_target[1], mode="nearest")
        if dtype == torch.bool:
            _mask = _mask.bool()
        _mask = _mask.squeeze()
    if len(mask.size()) == 2:
        if len(target_size) == 4:
            _mask = _mask.unsqueeze(-1)
        if len(target_size) == 5:
            _mask = _mask.unsqueeze(-1)
            _mask = _mask.unsqueeze(-1)
        _mask = _mask.expand(_target)
        _mask = _mask.unsqueeze(1)
    elif len(mask.size()) == 3:
        if len(target_size) == 5:
            _mask = _mask.unsqueeze(-1)
        _mask = _mask.expand(_target)
        _mask = _mask.unsqueeze(1)
    else:
        return _mask
    return _mask


class _MaskedBatchNormNd(nn.Module):

    __constants__ = ["filter_maskedval", "mode", "last_gamma",
                     "track_running_stats", "momentum", "eps", "num_features"]
    num_features: int
    eps: float
    momentum: float
    track_running_stats: bool
    last_gamma: bool
    mode: list
    filter_maskedval: bool

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 last_gamma=False,
                 filter_maskedval=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.last_gamma = last_gamma
        self.filter_maskedval = filter_maskedval

        # pylint doesn't judge register_buffer() as initialization.
        # And "self.num_batches_tracked = None" disturbs register_buffer().
        # https://github.com/pytorch/pytorch/issues/1874
        # So, it is difficult to remove no-member error of num_batches_tracked
        # correctly.

    def _reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        # num_batches_tracked should be initialized in derived classes.
        # pylint: disable=no-member
        self.num_batches_tracked.zero_()
        if self.last_gamma:
            nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    # pylint: disable=unused-argument, no-self-use
    def _check_input_dim(self, feature):
        message = "_MaskedBatchNormNd can not be called directory. " \
            "Please use derived classes."
        raise NotImplementedError(message)

    def _initialize_mask(self, feature, mask=None):
        size = feature.size()
        if mask is None:
            _mask = torch.ones(list(size[0:1]) + [1] + list(size[2:]),
                               dtype=feature.dtype, device=feature.device,
                               requires_grad=False)
        else:
            _mask = mask.to(feature.dtype)
        self._check_input_dim(_mask)

        num = torch.clamp(_mask.sum(), min=1)
        _mask_bn = _mask / num
        _mask_bn = _mask_bn.expand(size)

        _mask_one = _mask.expand(size)
        return _mask_bn, _mask_one

    def forward(self, feature, mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, C, *]`, * indicate 1-3 axes.
          - mask: `[N, 1, *]`

        # Returns:
          - feature: `[N, C, *]`
        """
        self._check_input_dim(feature)
        size = feature.size()
        _mask_bn, _mask_one = self._initialize_mask(feature, mask)
        if len(size) == 3:
            axes_bn = [0, 2]
        elif len(size) == 4:
            axes_bn = [0, 2, 3]
        elif len(size) == 5:
            axes_bn = [0, 2, 3, 4]

        if self.momentum is not None:
            exponential_average_factor = self.momentum
        elif self.training and self.track_running_stats:
            # num_batches_tracked should be initialized in derived classes.
            # pylint: disable=no-member
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1  # pylint: disable=no-member
                # use cumulative moving average
                # pylint: disable=no-member
                exponential_average_factor = 1.0 / \
                    float(self.num_batches_tracked)
        else:
            exponential_average_factor = 0.0

        mean_bn = (_mask_bn * feature).sum(axes_bn, keepdim=True)
        var_bn = (_mask_bn * (feature - mean_bn)
                  ** 2).sum(axes_bn, keepdim=True)
        var_bn = torch.clip(var_bn, min=0)
        assert (var_bn >= 0).all(
        ), f"{feature}, {_mask_bn}, {mean_bn}, {var_bn}"
        if self.training:
            with torch.no_grad():
                # running_mean and running_var should be initialized in
                # the derived classes by register_buffer().
                # pylint: disable=attribute-defined-outside-init
                self.running_mean = exponential_average_factor * mean_bn \
                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var_bn \
                    + (1 - exponential_average_factor) * self.running_var
        elif self.track_running_stats:
            mean_bn = self.running_mean
            var_bn = self.running_var
        mean_bn = mean_bn.expand(size)
        var_bn = var_bn.expand(size)

        outputs = (feature-mean_bn) / (var_bn+self.eps).sqrt()
        outputs = outputs.view(size)

        # Mask out.
        if self.filter_maskedval:
            outputs = outputs * _mask_one
        return outputs * self.weight + self.bias

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, " \
            "track_running_stats={track_running_stats}, " \
            "last_gamma={last_gamma}, " \
            "filter_maskedval={filter_maskedval}".format(**self.__dict__)


class MaskedBatchNorm1d(_MaskedBatchNormNd):
    """Masked switchable normalization for 1-dimensional features.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 last_gamma=False,
                 filter_maskedval=False):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         last_gamma=last_gamma,
                         filter_maskedval=filter_maskedval)
        # pylint error of torch.tensor() should be solved in the future
        # release.
        # https://github.com/pytorch/pytorch/issues/24807
        self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
        self.register_buffer("running_var", torch.ones(1, num_features, 1))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self._reset_parameters()

    def _check_input_dim(self, feature):
        if feature.dim() != 3:
            raise ValueError(f"expected 3D input (got {feature.dim()}D input)")


class MaskedBatchNorm2d(_MaskedBatchNormNd):
    """Masked switchable normalization for 2-dimensional features.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 last_gamma=False,
                 filter_maskedval=False):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         last_gamma=last_gamma,
                         filter_maskedval=filter_maskedval)

        # pylint error of torch.tensor() should be solved in the future
        # release.
        # https://github.com/pytorch/pytorch/issues/24807
        self.register_buffer(
            "running_mean", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self._reset_parameters()

    def _check_input_dim(self, feature):
        if feature.dim() != 4:
            raise ValueError(f"expected 4D input (got {feature.dim()}D input)")


class MaskedBatchNorm3d(_MaskedBatchNormNd):
    """Masked switchable normalization for 3-dimensional features.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 last_gamma=False,
                 filter_maskedval=False):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         last_gamma=last_gamma,
                         filter_maskedval=filter_maskedval)

        # pylint error of torch.tensor() should be solved in the future
        # release.
        # https://github.com/pytorch/pytorch/issues/24807
        self.register_buffer(
            "running_mean", torch.zeros(1, num_features, 1, 1, 1))
        self.register_buffer(
            "running_var", torch.ones(1, num_features, 1, 1, 1))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self._reset_parameters()

    def _check_input_dim(self, feature):
        if feature.dim() != 5:
            raise ValueError(f"expected 5D input (got {feature.dim()}D input)")


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
