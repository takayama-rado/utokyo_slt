#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_3_A1_STGCN_isolated_slr_mode: Add STGCN layer.
-------------------------------------------------------------------------------

# A permissive license whose main conditions require preservation of
# copyright and license notices.
# Contributors provide an express grant of patent rights.
# Licensed works, modifications, and larger works may be distributed under
# different terms and without source code.

Copyright (c) 2026 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import random

from argparse import (
    ArgumentParser,
    Namespace
)
from dataclasses import (
    dataclass,
    field
)
from datetime import datetime
from typing import Tuple

# Third party's modules
import numpy as np

import torch

from torch import nn
from torch.nn import CrossEntropyLoss

from torchvision.transforms import Compose

# Local modules
from modules.dataset import (
    make_wlasl_dataloader
)
from modules.model import (
    GPoolRecognitionHead
)
from modules.transforms import (
    PartsBasedNormalization,
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    ToTensor)


# Pose.
USE_POSE_REGULAR = np.arange(0, 23)
# Hands.
USE_LHAND_REGULAR = np.arange(33, 33+21)
USE_RHAND_REGULAR = np.arange(33+21, 33+21+21)

# Face.
USE_LIP_REGULAR = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
USE_NOSE_REGULAR = [98, 1, 327, 2]
USE_REYE_REGULAR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    246, 161, 160, 159, 158, 157, 173]
USE_LEYE_REGULAR = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                    466, 388, 387, 386, 385, 384, 398]
USE_FACE_REGULAR = np.array(
    USE_LIP_REGULAR + USE_NOSE_REGULAR + USE_REYE_REGULAR + USE_LEYE_REGULAR) \
    + 33 + 21 + 21


# Pose.
POSE_CONNECTIONS_REGULAR = np.array([
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
    [5, 6], [6, 8], [9, 10], [0, 9], [0, 10], [0, 11], [0, 12],
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19],
    [15, 21], [17, 19], [12, 14], [14, 16], [16, 18],
    [16, 20], [16, 22], [18, 20]])
# Hands.
HAND_CONNECTIONS_REGULAR = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]])
LHAND_CONNECTIONS_REGULAR = HAND_CONNECTIONS_REGULAR + 33
RHAND_CONNECTIONS_REGULAR = HAND_CONNECTIONS_REGULAR + 21 + 33

# Face.
LIP_CONNECTIONS_REGULAR = np.array([
    # Outer line.
    [0, 37], [37, 39], [39, 40], [40, 185], [185, 61],
    [61, 146], [146, 91], [91, 181], [181, 84], [84, 17],
    [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
    [291, 409], [409, 270], [270, 269], [269, 267], [267, 0],
    # Inner line.
    [13, 82], [82, 81], [81, 80], [80, 191], [191, 78],
    [78, 95], [95, 88], [88, 178], [178, 87], [87, 14],
    [317, 402], [402, 318], [318, 324], [324, 308], [308, 415],
    [415, 310], [310, 311], [311, 312], [312, 13],
    # Outer to inner.
    [61, 78], [291, 308]
])
NOSE_CONNECTIONS_REGULAR = np.array([
    [1, 98], [98, 2], [2, 327], [327, 1]
])
REYE_CONNECTIONS_REGULAR = np.array([
    [133, 173], [173, 157], [157, 158], [158, 159], [159, 160],
    [161, 246], [246, 33], [33, 7], [7, 163], [163, 144],
    [144, 145], [145, 153], [153, 154], [154, 155], [155, 133]
])
LEYE_CONNECTIONS_REGULAR = np.array([
    [362, 398], [398, 384], [384, 385], [385, 386], [386, 387],
    [387, 388], [388, 466], [466, 263], [263, 249], [249, 390],
    [390, 373], [373, 374], [374, 380], [380, 381], [381, 382],
    [382, 362]
])
FACE_INNER_CONNECTIONS_REGULAR = np.array([
    [1, 133], [1, 362], [2, 0]
])
FACE_CONNECTIONS_REGULAR = np.concatenate(
    [LIP_CONNECTIONS_REGULAR,
     NOSE_CONNECTIONS_REGULAR,
     REYE_CONNECTIONS_REGULAR,
     LEYE_CONNECTIONS_REGULAR,
     FACE_INNER_CONNECTIONS_REGULAR], axis=0) + 33 + 21 + 21


POSE_ORIGIN_REGULAR = [11, 12]  # Neck (between shoulder).
POSE_UNIT1_REGULAR = [11]  # LShoulder.
POSE_UNIT2_REGULAR = [12]  # RShoulder.

LHAND_ORIGIN_REGULAR = [0, 2, 5, 9, 13, 17]  # Center of parm.
LHAND_UNIT1_REGULAR = [0]  # Wrist.
LHAND_UNIT2_REGULAR = [2, 5, 9, 13, 17]  # Center of MCP.

RHAND_ORIGIN_REGULAR = [0, 2, 5, 9, 13, 17]  # Center of parm.
RHAND_UNIT1_REGULAR = [0]  # Wrist.
RHAND_UNIT2_REGULAR = [2, 5, 9, 13, 17]  # Center of MCP.

FACE_ORIGIN_REGULAR = [1, 2]  # Center of nose.
FACE_UNIT1_REGULAR = [7]  # Outer corner of right eye.
FACE_UNIT2_REGULAR = [42]  # Outer corner of left eye.


def get_hop_distance(num_node: int,
                     edge: Tuple[Tuple[int, int]],
                     max_hop: int = 1) -> np.ndarray:
    """ Get hop distances between nodes.

    # Args:
      - num_node: The number of nodes.
      - edge: Tuple to define edges. For example (0, 1) defines edge between
        node0 and node1.
      - max_hop: Maximum hop distance to be computed. inf is assigned if
        over this.
    # Returns:
      - hop_dis: Matrix defining hop distance between nodes.

    # Examples::
      >>> num_node = 3
      >>> edge = [[0, 1], [1, 2]]
      >>> get_hop_distance(num_node, edge, max_hop=1)
      >>> array([[ 0.,  1., inf],
                 [ 1.,  0.,  1.],
                 [inf,  1.,  0.]])
      >>> get_hop_distance(num_node, edge, max_hop=2)
      >>> array([[ 0.,  1.,  2.],
                 [ 1.,  0.,  1.],
                 [ 2.,  1.,  0.]])
    """
    adj_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        adj_mat[j, i] = 1
        adj_mat[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(
        adj_mat, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for dis in range(max_hop, -1, -1):
        hop_dis[arrive_mat[dis]] = dis
    return hop_dis


def calc_degree_div(adj_mat, div, eps, direction):
    """Calculate normalization factor based on the degree of adjacency matrix.
    """
    if direction == "col":
        diag = np.sum(adj_mat, 0) + eps
    else:
        diag = np.sum(adj_mat, 1) + eps
    num_node = adj_mat.shape[0]
    diag_normed = np.zeros((num_node, num_node))
    for i in range(num_node):
        if diag[i] > 0:
            diag_normed[i, i] = diag[i]**(div)
    return diag_normed


def normalize_undigraph(adj_mat, eps=0.0):
    """ Compute normalized undirected adjacency matrix.

    # Args:
      adj_mat: The adjacency matrix.
      eps: adj_mat small value to avoid empty row.
    # Returns:
      adj_normed: The normalized undirected adjacency matrix (DAD).
    """
    diag_normed_row = calc_degree_div(adj_mat, -0.5, eps, "row")
    diag_normed_col = calc_degree_div(adj_mat, -0.5, eps, "col")
    adj_normed = np.dot(np.dot(diag_normed_row, adj_mat), diag_normed_col)
    return adj_normed


@dataclass
class GraphSettings():
    """ Define graph structure.

    # Attributes:
      - max_hop: The maximum hop distance to define neighbors.
      - dilation: The dilation of joints in hop distance.
      - num_node: The number of joints.
      - strategy: The joint assignment strategy.
      - neighbor_link: The definition of connected joints.
      - center: The index of center joint.
    """
    max_hop: int = 1
    dilation: int = 1
    num_node: int = 8
    eps: float = 0.000

    neighbor_link: Tuple[Tuple[int, int], ...] = field(
        default_factory=lambda: ((4, 3), (3, 2), (7, 6), (6, 5),
                                 (2, 1), (5, 1), (1, 0)))
    centers: Tuple[int] = field(
        default_factory=lambda: (0,)
    )

    def __post_init__(self):
        self_link = [(i, i) for i in range(self.num_node)]
        self.edges = self_link + list(self.neighbor_link)

        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edges,  # type: ignore
                                        max_hop=self.num_node)
        self.adj_mat = self.get_adjacency(self.eps)

    def __str__(self):
        return f"{self.adj_mat}"

    def get_adjacency(self, eps=0.0):
        """Get adjacency matrix.
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        adj_mat = []
        _adj_mat = normalize_undigraph(adjacency, eps)
        for hop in valid_hop:
            a_equal = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for center in self.centers:
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        check0 = self.hop_dis[j, i] == hop
                        check1 = np.isfinite(self.hop_dis[i, center])
                        check2 = np.isfinite(self.hop_dis[j, center])

                        if check0 and check1 and check2:
                            dis_i = self.hop_dis[center, i]
                            dis_j = self.hop_dis[center, j]
                            if dis_i == dis_j:
                                a_equal[i, j] = _adj_mat[i, j]
                            elif dis_i > dis_j:
                                a_close[i, j] = _adj_mat[i, j]
                            else:
                                a_further[i, j] = _adj_mat[i, j]
            if hop == 0:
                adj_mat.append(a_equal)
            elif hop == 1:
                adj_mat.append(a_equal + a_close)
                adj_mat.append(a_further)
            else:
                adj_mat[1] += a_equal + a_close
                adj_mat[2] += a_further
        adj_mat = np.stack(adj_mat)
        return adj_mat


class SpatialGraphConv(nn.Module):
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
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self,
                feature: torch.Tensor,
                adj_mat: torch.Tensor) -> Tuple[torch.Tensor,
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

        feature = self.norm(feature)
        feature = self.activation(feature)

        return feature, adj_mat


class TemporalGraphConv(nn.Module):
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
                 stride: int,
                 dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=((kernel_size - 1) // 2, 0),
            dilation=(dilation, 1),
            groups=1,
            bias=True)

        self.norm = nn.BatchNorm2d(out_channels)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = Identity()

    def forward(self,
                feature: torch.Tensor) -> torch.Tensor:
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
        feature = self.norm(feature)
        feature = self.dropout(feature)
        return feature


class Zero(nn.Module):
    """Place holder layer to return zero vector.
    """
    # pylint: disable=unused-argument, no-self-use

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return 0


class Identity(nn.Module):
    """Place holder layer to return identity vector.
    """
    # pylint: disable=unused-argument, no-self-use

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return feature


class STGCNModule(nn.Module):
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

        self.sgcn = SpatialGraphConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=spatial_kernel_size)
        self.tgcn = TemporalGraphConv(in_channels=out_channels,
                                      out_channels=out_channels,
                                      dropout_rate=dropout_rate,
                                      kernel_size=temporal_kernel_size,
                                      stride=temporal_stride)

        if add_residual is False:
            self.residual = Zero()
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = Identity()
        else:
            self.residual = TemporalGraphConv(in_channels=in_channels,
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
        feature, adj_mat = self.sgcn(feature, adj_mat)
        feature = self.tgcn(feature) + res
        return self.activation(feature), adj_mat


class STGCNIsoSLR(nn.Module):
    def __init__(self,
                 graph_settings,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_stgcn_modules=1):
        super().__init__()

        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.register_buffer("adj_mat", adj_mat)

        self.front_norm = nn.BatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNModule(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNModule(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

        # Initialize parameters for edge importance weighting
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.adj_mat.shape))  # type: ignore
            for _ in self.stgcn_modules])

        self.recog_head = GPoolRecognitionHead(inter_channels, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, C, T, J]`
            - N: The batch size.
            - C: The channel dimension. C=2 for 2D joints, C=3 for 3D joints.
            - T: The temporal length.
            - J: The number of joints.
          - mask: `[N, T]`

        # Returns:
          - logits: `[N, num_classes]`
        """
        # ===================================================================
        # Apply front normalization.
        # ===================================================================
        # `[N, C, T, J] -> [N, J, C, T] -> [N, J*C, T]`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 3, 1, 2]).contiguous()
        feature = feature.view(N, J*C, T)
        feature = self.front_norm(feature)
        # `[N, J*C, T] -> [N, J, C, T] -> [N, C, T, J]`
        feature = feature.view(N, J, C, T)
        feature = feature.permute([0, 2, 3, 1])
        # ===================================================================
        # Main STGCN process.
        # ===================================================================
        for gcn, importance in zip(self.stgcn_modules, self.edge_importance):
            feature, _ = gcn(feature, self.adj_mat * importance,
                             feature_pad_mask)
        # ===================================================================
        # Generate logit.
        # ===================================================================
        logit = self.recog_head(feature, feature_pad_mask)
        return logit


def get_fullbody_landmarks():
    use_landmarks = np.concatenate([
        USE_POSE_REGULAR,
        USE_LHAND_REGULAR,
        USE_RHAND_REGULAR,
        USE_FACE_REGULAR])
    use_landmarks = np.sort(use_landmarks)
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks


def get_parts_connections(use_landmarks, partsname):
    connections = None
    if partsname == "pose":
        connections = np.array(
            [[pair[0], pair[1]] for pair in POSE_CONNECTIONS_REGULAR
             if pair[0] in use_landmarks and pair[1] in use_landmarks])
    elif partsname == "lhand":
        connections = np.array(
            [[pair[0], pair[1]] for pair in LHAND_CONNECTIONS_REGULAR
             if pair[0] in use_landmarks and pair[1] in use_landmarks])
    elif partsname == "rhand":
        connections = np.array(
            [[pair[0], pair[1]] for pair in RHAND_CONNECTIONS_REGULAR
             if pair[0] in use_landmarks and pair[1] in use_landmarks])
    elif partsname == "face":
        connections = np.array(
            [[pair[0], pair[1]] for pair in FACE_CONNECTIONS_REGULAR
             if pair[0] in use_landmarks and pair[1] in use_landmarks])
    assert connections is not None
    connections_sorted = np.array([[np.where(pair[0] == use_landmarks)[0][0],
                                    np.where(pair[1] == use_landmarks)[0][0]]
                                   for pair in connections])
    return connections, connections_sorted


def train_loop_iso(dataloader,
                   model,
                   loss_fn,
                   optimizer,
                   device,
                   use_mask=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    # Switch to training mode.
    model.train()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        tokens = batch_sample["tokens"]
        feature_pad_mask = batch_sample["feature_pad_mask"]

        # Predict and compute loss.
        feature = feature.to(device)
        tokens = tokens.to(device)
        if use_mask is False:
            feature_pad_mask = None
        else:
            feature_pad_mask = feature_pad_mask.to(device)

        pred = model(feature, feature_pad_mask=feature_pad_mask)
        loss = loss_fn(pred, tokens.squeeze(-1))

        # Back propagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Print current loss per 100 steps.
        if batch_idx % 10 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")
    train_loss /= num_batches
    print("Train performance: \n",
          f"Avg loss:{train_loss:>8f}\n")
    return train_loss


def val_loop_iso(dataloader, model, loss_fn, device, use_mask=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Switch to evaluation mode.
    model.eval()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            tokens = batch_sample["tokens"]
            feature_pad_mask = batch_sample["feature_pad_mask"]

            feature = feature.to(device)
            tokens = tokens.to(device)
            if use_mask is False:
                feature_pad_mask = None
            else:
                feature_pad_mask = feature_pad_mask.to(device)
            pred = model(feature, feature_pad_mask=feature_pad_mask)
            val_loss += loss_fn(pred, tokens.squeeze(-1)).item()
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    return val_loss


def test_loop_iso(dataloader, model, device, use_mask=False):
    size = len(dataloader.dataset)
    correct = 0

    # Switch to evaluation mode.
    model.eval()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            tokens = batch_sample["tokens"]
            feature_pad_mask = batch_sample["feature_pad_mask"]

            feature = feature.to(device)
            tokens = tokens.to(device)
            if use_mask is False:
                feature_pad_mask = None
            else:
                feature_pad_mask = feature_pad_mask.to(device)
            pred = model(feature, feature_pad_mask=feature_pad_mask)
            pred_ids = pred.argmax(dim=1).unsqueeze(-1)
            count = (pred_ids == tokens).sum().detach().cpu().numpy()
            correct += int(count)

    acc = correct / size * 100
    print("Test performance: \n",
          f"Accuracy:{acc:>0.1f}%")
    return acc


def save_checkpoint(ckpt_path, model,
                    epoch=0,
                    optimizer=None,
                    scheduler=None):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {}
    checkpoint = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_random"] = torch.cuda.get_rng_state()
        checkpoint["cuda_random_all"] = torch.cuda.get_rng_state_all()

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)


def main(args: Namespace, parser: ArgumentParser):
    try:
        data_dir = args.data_dir
        out_dir = args.out_dir
        batch_size = args.batch_size
        epochs = args.epochs
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    # Define models.
    # in_channels: J * C
    #   J: The number of joints (141).
    #   C: The number of elements in a joint (3: x, y, z).
    # out_channels: The number of words + keywords.
    dictionary = os.path.join(data_dir, "wlasl_100.json")

    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    in_channels = len(use_features)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

    # Define graphs.
    _, pose_connections = get_parts_connections(use_landmarks, "pose")
    _, lhand_connections = get_parts_connections(use_landmarks, "lhand")
    _, rhand_connections = get_parts_connections(use_landmarks, "rhand")
    _, face_connections = get_parts_connections(use_landmarks, "face")

    connections = np.concatenate([
        pose_connections,
        lhand_connections,
        rhand_connections,
        face_connections], axis=0)

    centers = [
        0,  # Nose
        len(USE_POSE_REGULAR),  # Left wrist
        len(USE_POSE_REGULAR) + len(USE_LHAND_REGULAR),  # Right wrist
        # The center of lip's outline.
        len(USE_POSE_REGULAR) + len(USE_LHAND_REGULAR) + len(USE_RHAND_REGULAR)
    ]

    graph_settings = GraphSettings(
        num_node=len(use_landmarks),
        neighbor_link=connections,  # type: ignore
        centers=centers)  # type: ignore

    model = STGCNIsoSLR(graph_settings,
                        in_channels,
                        out_channels,
                        inter_channels=64,
                        num_stgcn_modules=1)
    print(model)

    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero(),
        PartsBasedNormalization(
            pose_head=0,
            pose_num=len(USE_POSE_REGULAR),
            pose_origin=POSE_ORIGIN_REGULAR,
            pose_unit1=POSE_UNIT1_REGULAR,
            pose_unit2=POSE_UNIT2_REGULAR,
            lhand_head=len(USE_POSE_REGULAR),
            lhand_num=len(USE_LHAND_REGULAR),
            lhand_origin=LHAND_ORIGIN_REGULAR,
            lhand_unit1=LHAND_UNIT1_REGULAR,
            lhand_unit2=LHAND_UNIT2_REGULAR,
            rhand_head=len(USE_POSE_REGULAR) + len(USE_LHAND_REGULAR),
            rhand_num=len(USE_RHAND_REGULAR),
            rhand_origin=RHAND_ORIGIN_REGULAR,
            rhand_unit1=RHAND_UNIT1_REGULAR,
            rhand_unit2=RHAND_UNIT2_REGULAR,
            face_head=len(USE_POSE_REGULAR) +
            len(USE_LHAND_REGULAR) + len(USE_RHAND_REGULAR),
            face_num=len(USE_FACE_REGULAR),
            face_origin=FACE_ORIGIN_REGULAR,
            face_unit1=FACE_UNIT1_REGULAR,
            face_unit2=FACE_UNIT2_REGULAR),
        ToTensor()])

    track_dir = os.path.join(data_dir, "tracks", "train")
    label_dir = os.path.join(data_dir, "labels", "train")

    dataloader = make_wlasl_dataloader(track_dir, label_dir, dictionary,
                                       batch_size=2, shuffle=True,
                                       transforms=transforms)

    # --------------------------------------------------
    # Sanity check.
    # --------------------------------------------------
    # Fetch data.
    sample = next(iter(dataloader))
    tracks = sample["feature"]
    tokens = sample["tokens"]
    mask = sample["feature_pad_mask"]

    # Prediction.
    logits = model(tracks)
    print(logits.shape)
    probs = torch.softmax(logits, dim=1)
    print(probs.shape)

    # Predicted words.
    preds = probs.argmax(dim=1)
    print(preds.shape)
    print(preds)

    # Translate.
    token2word = {value["token"]: key for key, value in word_dict.items()}
    preds = preds.detach().cpu().numpy().flatten()
    tokens = tokens.detach().cpu().numpy().flatten()

    pred_words = [token2word[token] for token in preds]
    real_words = [token2word[token] for token in tokens]
    print(pred_words)
    print(real_words)

    # --------------------------------------------------
    # Train settings.
    # --------------------------------------------------
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using {device} for computation.")

    load_into_ram = True
    eval_every_n_epochs = 1

    train_dataloader = make_wlasl_dataloader(
        os.path.join(data_dir, "tracks", "train"),
        os.path.join(data_dir, "labels", "train"),
        dictionary,
        batch_size=batch_size,
        shuffle=True,
        load_into_ram=load_into_ram,
        transforms=transforms)
    val_dataloader = make_wlasl_dataloader(
        os.path.join(data_dir, "tracks", "val"),
        os.path.join(data_dir, "labels", "val"),
        dictionary,
        batch_size=batch_size,
        shuffle=False,
        load_into_ram=load_into_ram,
        transforms=transforms)
    test_dataloader = make_wlasl_dataloader(
        os.path.join(data_dir, "tracks", "test"),
        os.path.join(data_dir, "labels", "test"),
        dictionary,
        batch_size=1,
        shuffle=False,
        load_into_ram=load_into_ram,
        transforms=transforms)

    model.to(device)

    now = datetime.now()
    save_dir = os.path.join(
        out_dir,
        "A3_A1_wlasl_stgcn_isolated_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)

    # Save hyper-parameters.
    hparams = {
        "data_dir": data_dir,
        "dictionary": dictionary,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "load_into_ram": load_into_ram,
        "epochs": epochs,
    }

    with open(os.path.join(save_dir, "hparams.json"), "w") as fwrite:
        json.dump(hparams, fwrite, indent=4)

    # --------------------------------------------------
    # Training loop.
    # --------------------------------------------------
    # Training.
    print("Start training...")
    max_acc = 0
    min_val = 100000
    max_epoch = 0
    accuracies = {}
    train_losses = {}
    val_losses = {}
    for t in range(epochs):
        print("-" * 80)
        print(f"Epoch {t+1}")
        train_loss = train_loop_iso(
            train_dataloader, model, loss_fn, optimizer, device,
            use_mask=True)
        val_loss = val_loop_iso(val_dataloader, model,
                                loss_fn, device, use_mask=True)
        train_losses[t] = train_loss
        val_losses[t] = val_loss
        # Save checkpoint having minimum validation loss.
        if min_val > val_loss:
            ckpt_path = os.path.join(save_dir, "best_val_loss.ckpt")
            save_checkpoint(ckpt_path, model, t, optimizer)
            min_val = val_loss

        if (t+1) % eval_every_n_epochs == 0:
            acc = test_loop_iso(test_dataloader, model, device, use_mask=True)
            accuracies[t] = acc
            if acc > max_acc:
                max_acc = acc
                max_epoch = t
                # Save checkpoint having maximum accuracy.
                ckpt_path = os.path.join(save_dir, "best_acc.ckpt")
                save_checkpoint(ckpt_path, model, t, optimizer)

    print(f"The maximum accuracy is {max_acc} at {max_epoch}th epoch.")

    # Save final checkpoint.
    ckpt_path = os.path.join(save_dir, "last.ckpt")
    save_checkpoint(ckpt_path, model, epochs, optimizer)

    # Save results.
    with open(os.path.join(save_dir, "train_losses.json"), "w") as fwrite:
        json.dump(train_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "val_losses.json"), "w") as fwrite:
        json.dump(val_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "accuracies.json"), "w") as fwrite:
        json.dump(accuracies, fwrite, indent=4)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--out-dir", type=str, default="result")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)

    return parser


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)

    # Parse arguments.
    PARSER = set_argparse()
    ARGS = PARSER.parse_args()
    retval = main(ARGS, PARSER)
    PARSER.exit(retval, "Terminate {0} with status {1}.\n".format(
        os.path.basename(__file__), retval))
