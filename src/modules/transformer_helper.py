#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""transformer: The helper functions/classes for Transformer.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import (
    Optional)

# Third party's modules
import torch
from torch import nn

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def make_san_mask(pad_mask: torch.Tensor,
                  causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Make self-attention mask.

    # Args:
      - pad_mask: The padding mask. `[B, T]`
      - causal_mask: `[B, T (query), T (key)]`
        The mask for future context. For example, if T = 3, causal_mask
        should be:
        [[1, 0, 0,
         [1, 1, 0,
         [1, 1, 1]]
    # Returns:
      - san_mask: `[B, T (query), T (key)]`
    """
    # xx_mask: `[B, qlen, klen]`
    san_mask = pad_mask.unsqueeze(1).repeat([1, pad_mask.shape[-1], 1])
    if causal_mask is not None:
        san_mask = san_mask & causal_mask
    return san_mask


def make_causal_mask(ref_mask: torch.Tensor, lookahead: int = 0):
    """Make causal mask.
    # Args:
      - ref_mask: `[B, T (query)]`
        The reference mask to make causal mask.
      - lookahead: lookahead frame
    # Returns:
      - ref_mask: `[T (query), T (key)]`
    """
    causal_mask = ref_mask.new_ones([ref_mask.size(1), ref_mask.size(1)],
                                    dtype=ref_mask.dtype)
    causal_mask = torch.tril(causal_mask,
                             diagonal=lookahead,
                             out=causal_mask).unsqueeze(0)
    return causal_mask


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.
    # Args:
      - dim_model (int): dimension of MultiHeadAttentionMechanism.
      - dropout (float): dropout probability
      - max_len (int): maximum length for sinusoidal positional encoding
    """

    def __init__(self,
                 dim_model: int,
                 dropout: float,
                 max_len: int = 5000):

        super().__init__()

        self.dim_model = dim_model

        # Compute the positional encodings once in log space.
        pose = torch.zeros(max_len, dim_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float()
                             * -(math.log(10000.0) / dim_model))
        pose[:, 0::2] = torch.sin(position * div_term)
        pose[:, 1::2] = torch.cos(position * div_term)
        pose = pose.unsqueeze(0)  # for batch dimension
        self.register_buffer("pose", pose)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                feature):
        """Perform forward computation.

        # Args:
          - feature (FloatTensor): `[B, T, dim_model]`

        # Returns:
          - feature (FloatTensor): `[B, T, dim_model]`
        """
        feature = feature + self.pose[:, :feature.shape[1]]
        feature = self.dropout(feature)
        return feature


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
