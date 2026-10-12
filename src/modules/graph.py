#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""graph: Skeleton graph for GCN modules.
-------------------------------------------------------------------------------

This script is modified from:
https://github.com/open-mmlab/mmskeleton
Original copyright is as follows:

# ------------------------------------------------------------------------------
# Copyright (c) Open-MMLab, Multimedia Laboratory, CUHK.
# open-mmlab/mmskeleton is licensed under the
# Apache License 2.0
# A permissive license whose main conditions require preservation of
# copyright and license notices.
# Contributors provide an express grant of patent rights.
# Licensed works, modifications, and larger works may be distributed under
# different terms and without source code.
#
# License is placed on:
# https://github.com/open-mmlab/mmskeleton/blob/master/LICENSE
# ------------------------------------------------------------------------------

For the details, please see https://arxiv.org/abs/1801.07455 .

Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dataclasses import (
    dataclass,
    field)
from typing import (
    Tuple)

# Third party's modules
import numpy as np

# Local modules


# Execution settings
VERSION = "%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


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
                                        self.edges,
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


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
