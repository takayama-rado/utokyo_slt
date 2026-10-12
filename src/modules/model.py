#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""model: Model definition.
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

# Third party's modules
import torch
from torch import nn
from torch.nn import functional as F

# Local modules
from modules.maskednorm import (
    MaskedBatchNorm1d,
    insert_axes_to_mask)
from modules.stgcn import (
    STGCNModule,
    STGCNPPModule)
from modules.stgcn_wm import (
    STGCNModuleWM,
    STGCNPPModuleWM)
from modules.transformer_helper import (
    PositionalEncoding)

# Execution settings
VERSION = "%(prog)s dev"


class GPoolRecognitionHead(nn.Module):
    """Recognition head with global average pooling.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head = nn.Linear(in_channels, out_channels)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.head.weight,
                        0,
                        math.sqrt(1. / self.out_channels))

    def forward(self, feature, mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, C, T]` or `[N, C, T, J]`
            - N: The batch size.
            - C: The channel dimension.
            - T: The temporal length.
            - J: The number of joints.
          - mask: `[N, T]`

        # Returns:
          - logits: `[N, out_channels]`
        """
        # `[N, C, T] -> [N, C, T, 1]`
        if len(feature.shape) == 3:
            feature = feature.unsqueeze(-1)
        # Apply global average pooling.
        if mask is None:
            # Averaging over temporal and joint axes.
            feature = F.avg_pool2d(feature, feature.shape[2:])
        # Apply masked global average pooling.
        else:
            # `[N, T] -> [N, 1, T, 1]`
            _mask = mask[:, None, :, None]
            num = _mask.sum([2, 3])
            _mask = _mask / num[:, :, None, None]
            _mask = _mask.expand(feature.shape)
            # Now, _mask is converted to probability.
            # So, global average can be computed by the sum-product.
            feature = (_mask * feature).sum([2, 3], keepdim=True)
        # [N, C]
        feature = feature.view(feature.shape[0], -1)

        # Predict.
        feature = self.head(feature)
        return feature


class SimpleIsoSLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, 64)
        self.activation = nn.ReLU()
        self.head = GPoolRecognitionHead(64, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        """Forward computation.

        # Args:
          - inputs: `[N, in_channels, T, J]`
          - pad_mask: `[N, T]`
        # Returns:
          - logits: `[N, out_channels]`
        """
        # `[N, C, T, J] -> [N, T, J, C] -> [N, T, J*C]`
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.linear(feature)
        feature = self.activation(feature)

        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        logits = self.head(feature, feature_pad_mask)
        return logits


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
            nn.Parameter(torch.ones(self.adj_mat.shape))
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


class STGCNPPIsoSLR(nn.Module):
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
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = nn.BatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCNPP modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

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
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Generate logit.
        # ===================================================================
        logit = self.recog_head(feature, feature_pad_mask)
        return logit


class STGCNWMIsoSLR(nn.Module):
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

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1],
                                            filter_maskedval=True)

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNModuleWM(
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
            nn.Parameter(torch.ones(self.adj_mat.shape))
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
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)

        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
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


class STGCNPPWMIsoSLR(nn.Module):
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
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1],
                                            filter_maskedval=True)

        # Create STGCNPP modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

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
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)

        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
        # `[N, J*C, T] -> [N, J, C, T] -> [N, C, T, J]`
        feature = feature.view(N, J, C, T)
        feature = feature.permute([0, 2, 3, 1])
        # ===================================================================
        # Main STGCN process.
        # ===================================================================
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Generate logit.
        # ===================================================================
        logit = self.recog_head(feature, feature_pad_mask)
        return logit


class TransformerIsoBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _build_feature_extractor(self, in_channels, inter_channels):
        message = "This method must be implemented in a inherited class."
        raise NotImplementedError(message)

    def _build_tr_encoder(self,
                          inter_channels,
                          num_head,
                          num_encoder_layers,
                          dim_feedforward,
                          dropout_rate,
                          batch_first,
                          norm_first):
        self.pe_en = PositionalEncoding(dim_model=inter_channels,
                                        dropout=dropout_rate)

        enlayer = nn.TransformerEncoderLayer(d_model=inter_channels,
                                             nhead=num_head,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout_rate,
                                             batch_first=batch_first,
                                             norm_first=norm_first)
        tail_norm = nn.LayerNorm(inter_channels) if norm_first else None
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=num_encoder_layers,
            norm=tail_norm,
            enable_nested_tensor=not norm_first)

    def _build_recog_head(self,
                          inter_channels,
                          out_channels):
        self.recog_head = GPoolRecognitionHead(inter_channels, out_channels)

    def _invert_mask(self, mask):
        _mask = torch.logical_not(mask) if mask is not None else None
        return _mask

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        message = "This method must be implemented in a inherited class."
        raise NotImplementedError(message)

    def _apply_trencoder(self,
                         feature,
                         features_causal_mask=None,
                         feature_pad_mask=None):
        feature = self.pe_en(feature)

        # Note:
        # Different from the other layers, the attention of pytorch's
        # Transformer attends to `False` position indicated by the input masks.
        # So, we invert the flags of the input masks here.
        _features_causal_mask = self._invert_mask(features_causal_mask)
        _feature_pad_mask = self._invert_mask(feature_pad_mask)
        feature = self.tr_encoder(src=feature,
                                  mask=_features_causal_mask,
                                  src_key_padding_mask=_feature_pad_mask)
        return feature

    def _encoding(self,
                  feature,
                  feature_causal_mask=None,
                  feature_pad_mask=None):
        feature = self._apply_fext(feature, feature_pad_mask)
        feature = self._apply_trencoder(feature, feature_causal_mask,
                                        feature_pad_mask)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        return feature

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):

        enc_feature = self._encoding(feature,
                                     feature_causal_mask,
                                     feature_pad_mask)

        logits = self.recog_head(enc_feature, feature_pad_mask)
        return logits


class TransformerIsoSLR(TransformerIsoBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_head=2,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()

        self._build_feature_extractor(
            in_channels=in_channels,
            out_channels=inter_channels)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)
        self._build_recog_head(
            inter_channels=inter_channels,
            out_channels=out_channels)

    def _build_feature_extractor(self, in_channels, out_channels):
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # `[N, C, T, J] -> [N, T, J, C] -> [N, T, J*C]`
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape(
            [feature.shape[0], feature.shape[1], -1])
        feature = self.linear(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.norm(feature)
        feature = self.activation(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNTransformerIsoSLR(TransformerIsoBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 inter_channels=64,
                 num_head=2,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()

        self._build_feature_extractor(
            in_channels=in_channels,
            inter_channels=inter_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)
        self._build_recog_head(
            inter_channels=inter_channels,
            out_channels=out_channels)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
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
            nn.Parameter(torch.ones(self.adj_mat.shape))
            for _ in self.stgcn_modules])

        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = nn.BatchNorm1d(inter_channels)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
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
        # Convert for Transformer.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNPPTransformerIsoSLR(TransformerIsoBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 inter_channels=64,
                 num_head=2,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()

        self._build_feature_extractor(
            in_channels=in_channels,
            inter_channels=inter_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)
        self._build_recog_head(
            inter_channels=inter_channels,
            out_channels=out_channels)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = nn.BatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = nn.BatchNorm1d(inter_channels)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
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
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Convert for Transformer.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNWMTransformerIsoSLR(TransformerIsoBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 inter_channels=64,
                 num_head=2,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()

        self._build_feature_extractor(
            in_channels=in_channels,
            inter_channels=inter_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)
        self._build_recog_head(
            inter_channels=inter_channels,
            out_channels=out_channels)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.register_buffer("adj_mat", adj_mat)

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1],
                                            filter_maskedval=True)

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNModuleWM(
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
            nn.Parameter(torch.ones(self.adj_mat.shape))
            for _ in self.stgcn_modules])

        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = MaskedBatchNorm1d(inter_channels,
                                             filter_maskedval=True)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # ===================================================================
        # Apply front normalization.
        # ===================================================================
        # `[N, C, T, J] -> [N, J, C, T] -> [N, J*C, T]`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 3, 1, 2]).contiguous()
        feature = feature.view(N, J*C, T)
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)
        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
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
        # Convert for Transformer.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature, _mask)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNPPWMTransformerIsoSLR(TransformerIsoBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 inter_channels=64,
                 num_head=2,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()

        self._build_feature_extractor(
            in_channels=in_channels,
            inter_channels=inter_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)
        self._build_recog_head(
            inter_channels=inter_channels,
            out_channels=out_channels)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1],
                                            filter_maskedval=True)

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = MaskedBatchNorm1d(inter_channels,
                                             filter_maskedval=True)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # ===================================================================
        # Apply front normalization.
        # ===================================================================
        # `[N, C, T, J] -> [N, J, C, T] -> [N, J*C, T]`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 3, 1, 2]).contiguous()
        feature = feature.view(N, J*C, T)
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)
        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
        # `[N, J*C, T] -> [N, J, C, T] -> [N, C, T, J]`
        feature = feature.view(N, J, C, T)
        feature = feature.permute([0, 2, 3, 1])
        # ===================================================================
        # Main STGCN process.
        # ===================================================================
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Convert for Transformer.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature, _mask)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


# ===================================================================
# CSLR model.
# ===================================================================
class TransformerConBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _build_feature_extractor(self, in_channels, inter_channels):
        message = "This method must be implemented in a inherited class."
        raise NotImplementedError(message)

    def _build_tr_encoder(self,
                          inter_channels,
                          num_head,
                          num_encoder_layers,
                          dim_feedforward,
                          dropout_rate,
                          batch_first,
                          norm_first):
        self.pe_en = PositionalEncoding(dim_model=inter_channels,
                                        dropout=dropout_rate)

        enlayer = nn.TransformerEncoderLayer(d_model=inter_channels,
                                             nhead=num_head,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout_rate,
                                             batch_first=batch_first,
                                             norm_first=norm_first)
        tail_norm = nn.LayerNorm(inter_channels) if norm_first else None
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=num_encoder_layers,
            norm=tail_norm,
            enable_nested_tensor=not norm_first)

    def _build_tr_decoder(self,
                          inter_channels,
                          out_channels,
                          num_head,
                          num_decoder_layers,
                          dim_feedforward,
                          dropout_rate,
                          batch_first,
                          norm_first,
                          padding_idx):
        self.word_embed = nn.Embedding(num_embeddings=out_channels,
                                       embedding_dim=inter_channels,
                                       padding_idx=padding_idx)
        self.scale_wemb = math.sqrt(out_channels)

        self.pe_de = PositionalEncoding(dim_model=inter_channels,
                                        dropout=dropout_rate)

        delayer = nn.TransformerDecoderLayer(d_model=inter_channels,
                                             nhead=num_head,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout_rate,
                                             batch_first=batch_first,
                                             norm_first=norm_first)
        tail_norm = nn.LayerNorm(inter_channels) if norm_first else None
        self.tr_decoder = nn.TransformerDecoder(
            decoder_layer=delayer,
            num_layers=num_decoder_layers,
            norm=tail_norm)
        self.recog_head = nn.Linear(inter_channels, out_channels)
        nn.init.normal_(self.recog_head.weight,
                        0,
                        math.sqrt(1. / out_channels))

    def _invert_mask(self, mask):
        _mask = torch.logical_not(mask) if mask is not None else None
        return _mask

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        message = "This method must be implemented in a inherited class."
        raise NotImplementedError(message)

    def _apply_trencoder(self,
                         feature,
                         feature_causal_mask=None,
                         feature_pad_mask=None):
        feature = self.pe_en(feature)

        # Note:
        # Different from the other layers, the attention of pytorch's
        # Transformer attends to `False` position indicated by the input masks.
        # So, we invert the flags of the input masks here.
        _feature_causal_mask = self._invert_mask(feature_causal_mask)
        _feature_pad_mask = self._invert_mask(feature_pad_mask)
        feature = self.tr_encoder(src=feature,
                                  mask=_feature_causal_mask,
                                  src_key_padding_mask=_feature_pad_mask)
        return feature

    def _encoding(self,
                  feature,
                  feature_causal_mask=None,
                  feature_pad_mask=None):
        feature = self._apply_fext(feature, feature_pad_mask)
        feature = self._apply_trencoder(feature, feature_causal_mask,
                                        feature_pad_mask)
        return feature

    def _decoding(self,
                  enc_feature,
                  tokens,
                  feature_causal_mask=None,
                  feature_pad_mask=None,
                  tokens_causal_mask=None,
                  tokens_pad_mask=None):
        tgt_feature = self.word_embed(tokens) * self.scale_wemb
        tgt_feature = self.pe_de(tgt_feature)

        # Note:
        # Different from the other layers, the attention of pytorch's
        # Transformer attends to `False` position indicated by the input masks.
        # So, we invert the flags of the input masks here.
        _tokens_causal_mask = self._invert_mask(tokens_causal_mask)
        _feature_causal_mask = self._invert_mask(feature_causal_mask)
        _tokens_pad_mask = self._invert_mask(tokens_pad_mask)
        _feature_pad_mask = self._invert_mask(feature_pad_mask)
        tgt_feature = self.tr_decoder(
            tgt=tgt_feature,
            memory=enc_feature,
            tgt_mask=_tokens_causal_mask,
            memory_mask=_feature_causal_mask,
            tgt_key_padding_mask=_tokens_pad_mask,
            memory_key_padding_mask=_feature_pad_mask)
        outputs = self.recog_head(tgt_feature)
        return outputs

    def forward(self,
                feature,
                tokens,
                feature_causal_mask=None,
                feature_pad_mask=None,
                tokens_causal_mask=None,
                tokens_pad_mask=None):

        enc_feature = self._encoding(feature,
                                     feature_causal_mask,
                                     feature_pad_mask)
        outputs = self._decoding(enc_feature,
                                 tokens,
                                 feature_causal_mask,
                                 feature_pad_mask,
                                 tokens_causal_mask,
                                 tokens_pad_mask)
        return outputs

    def inference(self,
                  feature,
                  feature_pad_mask=None):
        enc_feature = self._encoding(feature,
                                     feature_pad_mask=feature_pad_mask)

        dec_inputs = torch.tensor([self.start_idx], dtype=torch.int64,
                                  device=feature.device).unsqueeze(1)
        for _ in range(self.max_length_outputs):
            preds = self._decoding(enc_feature,
                                   dec_inputs,
                                   feature_pad_mask=feature_pad_mask)
            # Extract last prediction.
            preds_ids = preds[:, -1].argmax(dim=-1)
            if len(preds_ids.shape) == 1:
                preds_ids = preds_ids.reshape([1, -1])
            dec_inputs = torch.cat([dec_inputs, preds_ids], dim=-1)
            if int(preds_ids[0, 0].detach().cpu().numpy()) == self.end_idx:
                break
        return dec_inputs


class TransformerConSLR(TransformerConBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding_idx=0,
                 start_idx=1,
                 end_idx=2,
                 max_length_outputs=20,
                 inter_channels=64,
                 num_encoder_head=2,
                 num_encoder_layers=2,
                 dim_feedforward_encoder=256,
                 num_decoder_head=2,
                 num_decoder_layers=2,
                 dim_feedforward_decoder=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters.
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length_outputs = max_length_outputs

        # ====================================================================
        # Define encoder.
        # ====================================================================
        self._build_feature_extractor(in_channels=in_channels,
                                      inter_channels=inter_channels)
        self._build_tr_encoder(
            inter_channels=inter_channels,
            num_head=num_encoder_head,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward_encoder,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first)

        # ====================================================================
        # Define decoder.
        # ====================================================================
        self._build_tr_decoder(
            inter_channels=inter_channels,
            out_channels=out_channels,
            num_head=num_decoder_head,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_decoder,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            norm_first=norm_first,
            padding_idx=padding_idx)

    def _build_feature_extractor(self, in_channels, inter_channels):
        self.linear = nn.Linear(in_channels, inter_channels)
        self.norm = nn.BatchNorm1d(inter_channels)
        self.activation = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # `[N, C, T, J] -> [N, T, J, C] -> [N, T, J*C]`
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape(
            [feature.shape[0], feature.shape[1], -1])
        feature = self.linear(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.norm(feature)
        feature = self.activation(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNTransformerConSLR(TransformerConBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 padding_idx=0,
                 start_idx=1,
                 end_idx=2,
                 max_length_outputs=20,
                 inter_channels=64,
                 num_encoder_head=2,
                 num_encoder_layers=2,
                 dim_feedforward_encoder=256,
                 num_decoder_head=2,
                 num_decoder_layers=2,
                 dim_feedforward_decoder=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters.
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length_outputs = max_length_outputs

        # ====================================================================
        # Define encoder.
        # ====================================================================
        self._build_feature_extractor(in_channels=in_channels,
                                      inter_channels=inter_channels,
                                      graph_settings=graph_settings,
                                      num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(inter_channels=inter_channels,
                               num_head=num_encoder_head,
                               num_encoder_layers=num_encoder_layers,
                               dim_feedforward=dim_feedforward_encoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first)

        # ====================================================================
        # Define decoder.
        # ====================================================================
        self._build_tr_decoder(inter_channels=inter_channels,
                               out_channels=out_channels,
                               num_head=num_decoder_head,
                               num_decoder_layers=num_decoder_layers,
                               dim_feedforward=dim_feedforward_decoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first,
                               padding_idx=padding_idx)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
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
            nn.Parameter(torch.ones(self.adj_mat.shape))
            for _ in self.stgcn_modules])

        # Build bridge layers.
        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = nn.BatchNorm1d(inter_channels)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
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
            feature, _ = gcn(feature, self.adj_mat *
                             importance, feature_pad_mask)
        # ===================================================================
        # Apply bridge layers.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNPPTransformerConSLR(TransformerConBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 padding_idx=0,
                 start_idx=1,
                 end_idx=2,
                 max_length_outputs=20,
                 inter_channels=64,
                 num_encoder_head=2,
                 num_encoder_layers=2,
                 dim_feedforward_encoder=256,
                 num_decoder_head=2,
                 num_decoder_layers=2,
                 dim_feedforward_decoder=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters.
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length_outputs = max_length_outputs

        # ====================================================================
        # Define encoder.
        # ====================================================================
        self._build_feature_extractor(in_channels=in_channels,
                                      inter_channels=inter_channels,
                                      graph_settings=graph_settings,
                                      num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(inter_channels=inter_channels,
                               num_head=num_encoder_head,
                               num_encoder_layers=num_encoder_layers,
                               dim_feedforward=dim_feedforward_encoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first)

        # ====================================================================
        # Define decoder.
        # ====================================================================
        self._build_tr_decoder(inter_channels=inter_channels,
                               out_channels=out_channels,
                               num_head=num_decoder_head,
                               num_decoder_layers=num_decoder_layers,
                               dim_feedforward=dim_feedforward_decoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first,
                               padding_idx=padding_idx)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = nn.BatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModule(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

        # Build bridge layers.
        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = nn.BatchNorm1d(inter_channels)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
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
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Apply bridge layers.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNWMTransformerConSLR(TransformerConBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 padding_idx=0,
                 start_idx=1,
                 end_idx=2,
                 max_length_outputs=20,
                 inter_channels=64,
                 num_encoder_head=2,
                 num_encoder_layers=2,
                 dim_feedforward_encoder=256,
                 num_decoder_head=2,
                 num_decoder_layers=2,
                 dim_feedforward_decoder=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters.
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length_outputs = max_length_outputs

        # ====================================================================
        # Define encoder.
        # ====================================================================
        self._build_feature_extractor(in_channels=in_channels,
                                      inter_channels=inter_channels,
                                      graph_settings=graph_settings,
                                      num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(inter_channels=inter_channels,
                               num_head=num_encoder_head,
                               num_encoder_layers=num_encoder_layers,
                               dim_feedforward=dim_feedforward_encoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first)

        # ====================================================================
        # Define decoder.
        # ====================================================================
        self._build_tr_decoder(inter_channels=inter_channels,
                               out_channels=out_channels,
                               num_head=num_decoder_head,
                               num_decoder_layers=num_decoder_layers,
                               dim_feedforward=dim_feedforward_decoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first,
                               padding_idx=padding_idx)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.register_buffer("adj_mat", adj_mat)

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNModuleWM(
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
            nn.Parameter(torch.ones(self.adj_mat.shape))
            for _ in self.stgcn_modules])

        # Build bridge layers.
        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = MaskedBatchNorm1d(inter_channels,
                                             filter_maskedval=True)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # ===================================================================
        # Apply front normalization.
        # ===================================================================
        # `[N, C, T, J] -> [N, J, C, T] -> [N, J*C, T]`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 3, 1, 2]).contiguous()
        feature = feature.view(N, J*C, T)
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)
        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
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
        # Apply bridge layers.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature, _mask)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


class STGCNPPWMTransformerConSLR(TransformerConBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_settings,
                 num_stgcn_modules=1,
                 padding_idx=0,
                 start_idx=1,
                 end_idx=2,
                 max_length_outputs=20,
                 inter_channels=64,
                 num_encoder_head=2,
                 num_encoder_layers=2,
                 dim_feedforward_encoder=256,
                 num_decoder_head=2,
                 num_decoder_layers=2,
                 dim_feedforward_decoder=256,
                 batch_first=True,
                 norm_first=True,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters.
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length_outputs = max_length_outputs

        # ====================================================================
        # Define encoder.
        # ====================================================================
        self._build_feature_extractor(in_channels=in_channels,
                                      inter_channels=inter_channels,
                                      graph_settings=graph_settings,
                                      num_stgcn_modules=num_stgcn_modules)
        self._build_tr_encoder(inter_channels=inter_channels,
                               num_head=num_encoder_head,
                               num_encoder_layers=num_encoder_layers,
                               dim_feedforward=dim_feedforward_encoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first)

        # ====================================================================
        # Define decoder.
        # ====================================================================
        self._build_tr_decoder(inter_channels=inter_channels,
                               out_channels=out_channels,
                               num_head=num_decoder_head,
                               num_decoder_layers=num_decoder_layers,
                               dim_feedforward=dim_feedforward_decoder,
                               dropout_rate=dropout_rate,
                               batch_first=batch_first,
                               norm_first=norm_first,
                               padding_idx=padding_idx)

    def _build_feature_extractor(self,
                                 in_channels,
                                 inter_channels,
                                 graph_settings,
                                 num_stgcn_modules):
        adj_mat = torch.tensor(graph_settings.adj_mat,
                               dtype=torch.float32,
                               requires_grad=False)
        self.adj_mat = nn.Parameter(adj_mat.clone())

        self.front_norm = MaskedBatchNorm1d(in_channels * adj_mat.shape[1])

        # Create STGCN modules.
        self.stgcn_modules = nn.ModuleList()
        for i in range(num_stgcn_modules):
            if i == 0:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=False
                ))
            else:
                self.stgcn_modules.append(STGCNPPModuleWM(
                    in_channels=inter_channels,
                    out_channels=inter_channels,
                    spatial_kernel_size=3,
                    temporal_kernel_size=9,
                    temporal_stride=1,
                    dropout_rate=0.1,
                    add_residual=True
                ))

        # Build bridge layers.
        self.bridge_layer = nn.Linear(inter_channels * graph_settings.num_node,
                                      inter_channels)
        self.bridge_norm = MaskedBatchNorm1d(inter_channels,
                                             filter_maskedval=True)
        self.bridge_act = nn.ReLU()

    def _apply_fext(self,
                    feature,
                    feature_pad_mask=None):
        # ===================================================================
        # Apply front normalization.
        # ===================================================================
        # `[N, C, T, J] -> [N, J, C, T] -> [N, J*C, T]`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 3, 1, 2]).contiguous()
        feature = feature.view(N, J*C, T)
        if feature_pad_mask is not None:
            _mask = insert_axes_to_mask(feature_pad_mask, feature.shape)
        else:
            _mask = feature_pad_mask
        feature = self.front_norm(feature, _mask)
        # `[N, J*C, T] -> [N, J, C, T] -> [N, C, T, J]`
        feature = feature.view(N, J, C, T)
        feature = feature.permute([0, 2, 3, 1])
        # ===================================================================
        # Main STGCN process.
        # ===================================================================
        for gcn in self.stgcn_modules:
            feature, _ = gcn(feature, self.adj_mat, feature_pad_mask)
        # ===================================================================
        # Apply bridge layers.
        # ===================================================================
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J]`
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.bridge_layer(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.bridge_norm(feature, _mask)
        feature = self.bridge_act(feature)
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        return feature


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
