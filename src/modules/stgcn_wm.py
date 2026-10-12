#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""stgcn: Spatial-Temporal GCN modules.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import (
    Any,
    Dict,
    List,
    Tuple)

# Third party's modules
import torch
from torch import nn

# Local modules
from modules.maskednorm import (
    MaskedBatchNorm2d,
    insert_axes_to_mask)
from modules.utils import (
    Identity,
    Zero)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class SpatialGraphConvWM(nn.Module):
    """The basic module to apply a spatial-graph convolution.

    # Attributes:
      - in_channels: The number of channels in the input.
      - out_channels: The number of channels produced by the convolution.
      - kernel_size: The size of the spatial-graph convolving kernel.

    # Note:
      For the details, please see https://arxiv.org/abs/1801.07455 .
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=True)
        self.norm = MaskedBatchNorm2d(out_channels, filter_maskedval=True)
        self.activation = nn.ReLU()

    def forward(self,
                feature: torch.Tensor,
                adj_mat: torch.Tensor,
                mask: torch.Tensor | None = None) -> Tuple[torch.Tensor,
                                                           torch.Tensor]:
        """Perform forward computation.

        # Args:
          - feature: `[N, in_channels, T, V]`
            - N: The batch size.
            - in_channels: The input channel dimension.
            - T: The temporal length.
            - V: The number of joints.
          - mask: `[N, T]`

        # Returns:
          - feature: `[N, out_channels, T, V]`
            - out_channels: The output channel dimension.
        """
        assert adj_mat.size(
            0) == self.kernel_size, f"{adj_mat.size(0)}:{self.kernel_size}"

        # Apply point-wise convolution.
        feature = self.conv(feature)

        n, kc, t, v = feature.size()  # pylint: disable=invalid-name
        # `[N, out_channels*kernel_size, t, v]
        # -> [N, kernel_size, out_channels, t, v]`
        feature = feature.view(n, self.kernel_size, kc //
                               self.kernel_size, t, v)
        # Sum over adjacency matrix.
        feature = torch.einsum('nkctv,kvw->nctw', (feature, adj_mat))
        feature = feature.contiguous()

        _mask = insert_axes_to_mask(mask, feature.shape) if mask is not None \
            else mask
        feature = self.norm(feature, _mask)
        feature = self.activation(feature)

        return feature, adj_mat


class TemporalGraphConvWM(nn.Module):
    """The basic module to apply a temporal-graph convolution.

    # Attributes:
      - in_channels: The number of channels in the input.
      - out_channels: The number of channels produced by the convolution.
      - dropout_rate: If dropout_rate > 0, the Dropout layer is inserted.
      - kernel_size: The size of the temporal-graph convolving kernel.
      - stride: The stride of the temporal-graph convolving kernel.
        If this value is greater than 1, the temporal length of the output
        sequence should be shorter than that of the input.

    # Note:
      For the details, please see https://arxiv.org/abs/1801.07455 .
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_rate: float,
                 kernel_size: int,
                 stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=((kernel_size - 1) // 2, 0),
            dilation=1,
            groups=1,
            bias=True)

        self.norm = MaskedBatchNorm2d(out_channels, filter_maskedval=True)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = Identity()

    def forward(self,
                feature: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        """Perform forward computation.

        # Args:
          - feature: `[N, in_channels, T, J]`
            - N: The batch size.
            - in_channels: The input channel dimension.
            - T: The temporal length.
            - V: The number of joints.

        # Returns:
          - feature: `[N, out_channels, T, J]`
            - out_channels: The output channel dimension.
        """
        feature = self.conv(feature)
        _mask = insert_axes_to_mask(mask, feature.shape) if mask is not None \
            else mask
        feature = self.norm(feature, _mask)
        feature = self.dropout(feature)
        return feature


class MultiStreamTemporalGraphConvWM(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_rate: float,
                 kernel_size: int,
                 stride: int,
                 cnn_branches_params: List[Dict[str, Any]] = [
                     {"type": "positional"},
                     {"type": "max"},
                     {"type": "tgcn",
                      "dilation": 1},
                     {"type": "tgcn",
                      "dilation": 2},
                     {"type": "tgcn",
                      "dilation": 3},
                     {"type": "tgcn",
                      "dilation": 4},
                 ]):
        super().__init__()
        inter_channels = out_channels // len(cnn_branches_params)
        res_inter_channels = out_channels % len(cnn_branches_params) \
            + inter_channels

        self.cnn_branches = nn.ModuleList()
        for i, param in enumerate(cnn_branches_params):
            branch_out_channel = res_inter_channels if i == 0 \
                else inter_channels
            front_cnn = nn.Conv2d(in_channels=in_channels,
                                  out_channels=branch_out_channel,
                                  kernel_size=(1, 1),
                                  stride=(stride, 1),
                                  padding=(0, 0),
                                  dilation=1,
                                  groups=1,
                                  bias=True)
            if param["type"] == "positional":
                norm = Identity()
                act = Identity()
                fext = Identity()
            elif param["type"] == "max":
                norm = MaskedBatchNorm2d(branch_out_channel,
                                         filter_maskedval=True)
                act = nn.ReLU()
                fext = nn.MaxPool2d(
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=((kernel_size - 1) // 2, 0))
            elif param["type"] == "tgcn":
                norm = MaskedBatchNorm2d(branch_out_channel,
                                         filter_maskedval=True)
                act = nn.ReLU()
                fext = nn.Conv2d(
                    in_channels=branch_out_channel,
                    out_channels=branch_out_channel,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(param["dilation"] *
                             (kernel_size - 1) // 2, 0),
                    dilation=(param["dilation"], 1),
                    groups=1,
                    bias=True)
            else:
                raise ValueError(f"Unknown CNN type: {param['type']}.")
            self.cnn_branches.append(
                nn.ModuleDict({
                    "front_cnn": front_cnn,
                    "norm": norm,
                    "act": act,
                    "fext": fext}))
        self.tail_conv = nn.ModuleDict({
            "norm": MaskedBatchNorm2d(out_channels, filter_maskedval=True),
            "act": nn.ReLU(),
            "fext": nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1)})

        self.norm = MaskedBatchNorm2d(out_channels, filter_maskedval=True)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = Identity()

    def forward(self,
                feature: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        """Perform forward computation.

        # Args:
          - feature: `[N, in_channels, T, J]`
            - N: The batch size.
            - in_channels: The input channel dimension.
            - T: The temporal length.
            - V: The number of joints.

        # Returns:
          - feature: `[N, out_channels, T, J]`
            - out_channels: The output channel dimension.
        """
        _mask = insert_axes_to_mask(mask, feature.shape) if mask is not None \
            else mask
        branch_features = []
        for cnn in self.cnn_branches:
            feature_b = cnn["front_cnn"](feature)
            feature_b = cnn["norm"](feature_b, _mask)
            feature_b = cnn["act"](feature_b)
            feature_b = cnn["fext"](feature_b)
            branch_features.append(feature_b)

        feature = torch.cat(branch_features, dim=1)
        feature = self.tail_conv["norm"](feature, _mask)
        feature = self.tail_conv["act"](feature)
        feature = self.tail_conv["fext"](feature)

        feature = self.norm(feature, _mask)
        feature = self.dropout(feature)
        return feature


class STGCNModuleWM(nn.Module):
    """ Spatial-Temporal GCN module.

    # Note:
      For the details, please see https://arxiv.org/abs/1801.07455 .
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_kernel_size,
                 temporal_kernel_size,
                 temporal_stride,
                 dropout_rate,
                 add_residual=False):
        super().__init__()

        self.sgcn = SpatialGraphConvWM(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=spatial_kernel_size)
        self.tgcn = TemporalGraphConvWM(in_channels=out_channels,
                                        out_channels=out_channels,
                                        dropout_rate=dropout_rate,
                                        kernel_size=temporal_kernel_size,
                                        stride=temporal_stride)

        if add_residual is False:
            self.residual = Zero()
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = Identity()
        else:
            self.residual = TemporalGraphConvWM(in_channels=in_channels,
                                                out_channels=out_channels,
                                                dropout_rate=0,
                                                kernel_size=1,
                                                stride=temporal_stride)
        self.activation = nn.ReLU()

    def forward(self, feature, adj_mat, mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, in_channels, t_in, V]`
            - N: The batch size.
            - in_channels: The input channel dimension.
            - t_in: The input temporal length.
            - V: The number of joints.
          - mask: `[N, t_in]`

        # Returns:
          - feature: `[N, out_channels, t_out, V]`
            - out_channels: The output channel dimension.
            - t_out: The output temporal length. If strided temporal
              convolution is applied. t_out should be less than t_in.
        """
        res = self.residual(feature, mask)
        feature, adj_mat = self.sgcn(feature, adj_mat, mask)
        feature = self.tgcn(feature, mask) + res
        return self.activation(feature), adj_mat


class STGCNPPModuleWM(nn.Module):
    """ Spatial-Temporal GCN module.

    # Note:
      For the details, please see https://arxiv.org/abs/1801.07455 .
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_kernel_size,
                 temporal_kernel_size,
                 temporal_stride,
                 dropout_rate,
                 add_residual=False):
        super().__init__()

        self.sgcn = SpatialGraphConvWM(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=spatial_kernel_size)
        # Multi stream (STGCN++).
        self.tgcn = MultiStreamTemporalGraphConvWM(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride)

        if add_residual is False:
            self.residual = Zero()
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = Identity()
        else:
            self.residual = TemporalGraphConvWM(in_channels=in_channels,
                                                out_channels=out_channels,
                                                dropout_rate=0,
                                                kernel_size=1,
                                                stride=temporal_stride)
        self.activation = nn.ReLU()

    def forward(self, feature, adj_mat, mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, in_channels, t_in, V]`
            - N: The batch size.
            - in_channels: The input channel dimension.
            - t_in: The input temporal length.
            - V: The number of joints.
          - mask: `[N, t_in]`

        # Returns:
          - feature: `[N, out_channels, t_out, V]`
            - out_channels: The output channel dimension.
            - t_out: The output temporal length. If strided temporal
              convolution is applied. t_out should be less than t_in.
        """
        res = self.residual(feature, mask)
        feature, adj_mat = self.sgcn(feature, adj_mat, mask)
        feature = self.tgcn(feature, mask) + res
        return self.activation(feature), adj_mat


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
