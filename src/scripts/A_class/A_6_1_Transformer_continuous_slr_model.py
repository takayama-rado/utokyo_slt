#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_6_1_Transformer_continuous_slr_model: Transformer CSLR model.
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
import math
import os

from argparse import (
    ArgumentParser,
    Namespace
)
from datetime import datetime

# Third party's modules
import numpy as np

import torch

from Levenshtein import distance

from torch import nn
from torch.nn import CrossEntropyLoss

from torchvision.transforms import Compose

# Local modules
from modules.dataset import (
    make_wlasl_dataloader
)
from modules.transformer_helper import (
    PositionalEncoding
)
from modules.defines import (
    get_calculation_landmarks,
    get_pose_landmarks,
    get_lhand_landmarks,
    get_rhand_landmarks,
    get_face_landmarks,
    get_fullbody_landmarks,
    get_fullbody_swappairs,
)
from modules.transforms import (
    RandomHorizontalFlip,
    RandomAffineTransform2D,
    RandomClip,
    RandomTimeWarping,
    PartsBasedNormalization,
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    ToTensor)
from modules.utils import save_checkpoint


def make_causal_mask(ref_mask, lookahead=0):
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

        en_layer = nn.TransformerEncoderLayer(d_model=inter_channels,
                                              nhead=num_head,
                                              dim_feedforward=dim_feedforward,
                                              dropout=dropout_rate,
                                              batch_first=batch_first,
                                              norm_first=norm_first)
        tail_norm = nn.LayerNorm(inter_channels) if norm_first else None
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=en_layer,
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
        for _ in range(self.max_length_outputs):  # type: ignore
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
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=inter_channels)
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


class InsertTokensForS2S():
    def __init__(self,
                 sos_token,
                 eos_token,
                 error_at_exist=False):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.error_at_exist = error_at_exist

    def check_format(self, tokens):
        insert_sos = False
        if tokens[0] != self.sos_token:
            insert_sos = True
        elif self.error_at_exist:
            message = f"The sos_token:{self.sos_token} is exist in {tokens}." \
                + "Please check the format."
            raise ValueError(message)
        insert_eos = False
        if tokens[-1] != self.eos_token:
            insert_eos = True
        elif self.error_at_exist:
            message = f"The eos_token:{self.eos_token} is exist in {tokens}." \
                + "Please check the format."
            raise ValueError(message)
        return insert_sos, insert_eos

    def __call__(self,
                 data):

        tokens = data["tokens"]
        dtype = tokens.dtype

        insert_sos, insert_eos = self.check_format(tokens)
        # Insert.
        new_tokens = []
        if insert_sos:
            new_tokens.append(self.sos_token)
        new_tokens += tokens.tolist()
        if insert_eos:
            new_tokens.append(self.eos_token)
        new_tokens = np.array(new_tokens, dtype=dtype)
        data["tokens"] = new_tokens
        return data


def train_loop_con(dataloader,
                   model,
                   loss_fn,
                   optimizer,
                   device,
                   use_mask=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    tokens_causal_mask = None

    # Switch to training mode.
    model.train()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        tokens = batch_sample["tokens"]
        feature_pad_mask = batch_sample["feature_pad_mask"]
        tokens_pad_mask = batch_sample["tokens_pad_mask"]
        initialize_mask = tokens_causal_mask is None or (
            tokens_causal_mask.shape[-1] != tokens_pad_mask.shape[-1])
        if initialize_mask:
            tokens_causal_mask = make_causal_mask(tokens_pad_mask)
            # `[1, L, L] -> [L, L]`
            tokens_causal_mask = tokens_causal_mask.squeeze(0)
            tokens_causal_mask = tokens_causal_mask.to(device)

        # Predict and compute loss.
        feature = feature.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)
        if use_mask is False:
            feature_pad_mask = None
        else:
            feature_pad_mask = feature_pad_mask.to(device)

        pred = model(feature=feature,
                     tokens=tokens,
                     feature_causal_mask=None,
                     feature_pad_mask=feature_pad_mask,
                     tokens_causal_mask=tokens_causal_mask,
                     tokens_pad_mask=tokens_pad_mask)

        # Calculate XE loss.
        # The temporal indices of pred and tokens are not aligned (shifted).
        # pred: `[N, L, C]`
        # tokens: `[N, L]`
        loss = 0
        for t in range(1, tokens.shape[-1]):
            loss += loss_fn(pred[:, t-1, :], tokens[:, t])
        loss /= (tokens.shape[-1] - 1)

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
    print("Training performance: \n",
          f"Avg loss:{train_loss:>8f}\n")
    return train_loss


def val_loop_con(dataloader, model, loss_fn, device, use_mask=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Switch to evaluation mode.
    model.eval()
    tokens_causal_mask = None
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            tokens = batch_sample["tokens"]
            feature_pad_mask = batch_sample["feature_pad_mask"]
            tokens_pad_mask = batch_sample["tokens_pad_mask"]
            initialize_mask = tokens_causal_mask is None or (
                tokens_causal_mask.shape[-1] != tokens_pad_mask.shape[-1])
            if initialize_mask:
                tokens_causal_mask = make_causal_mask(tokens_pad_mask)
                # `[1, L, L] -> [L, L]`
                tokens_causal_mask = tokens_causal_mask.squeeze(0)
                tokens_causal_mask = tokens_causal_mask.to(device)

            # Predict and compute loss.
            feature = feature.to(device)
            tokens = tokens.to(device)
            tokens_pad_mask = tokens_pad_mask.to(device)
            if use_mask is False:
                feature_pad_mask = None
            else:
                feature_pad_mask = feature_pad_mask.to(device)

            pred = model(feature=feature,
                         tokens=tokens,
                         feature_causal_mask=None,
                         feature_pad_mask=feature_pad_mask,
                         tokens_causal_mask=tokens_causal_mask,
                         tokens_pad_mask=tokens_pad_mask)
            # Calculate XE loss.
            # The temporal indices of pred and tokens are not aligned
            # (shifted).
            # pred: `[N, L, C]`
            # tokens: `[N, L]`
            loss = 0
            for t in range(1, tokens.shape[-1]):
                loss += loss_fn(pred[:, t-1, :], tokens[:, t])
            loss /= (tokens.shape[-1] - 1)
            val_loss += loss.item()
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    return val_loss


def test_loop_con(dataloader, model, device, use_mask=False,
                  ignore_ids=[0, 1, 2]):
    size = len(dataloader.dataset)
    wer = 0

    # Switch to evaluation mode.
    model.eval()
    with torch.no_grad():
        for i, batch_sample in enumerate(dataloader):
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

            pred_ids = model.inference(feature=feature,
                                       feature_pad_mask=feature_pad_mask)

            # Convert tokens to list or ids.
            tokens = tokens.detach().cpu().numpy().squeeze(0)
            pred_ids = pred_ids.detach().cpu().numpy().squeeze(0)
            tokens = [int(token)
                      for token in tokens if token not in ignore_ids]
            pred_ids = [int(pid) for pid in pred_ids if pid not in ignore_ids]
            if i == 0:
                print(f"{tokens}:{pred_ids}")
            length = max(len(tokens), len(pred_ids))
            wer += distance(pred_ids, tokens) / length

    awer = wer / size * 100
    print("Test performance: \n",
          f"Average WER:{awer:>0.1f}%")
    return awer


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

    parts_type = "compact"

    USE_POSE = get_pose_landmarks(parts_type)
    USE_LHAND = get_lhand_landmarks(parts_type)
    USE_RHAND = get_rhand_landmarks(parts_type)
    USE_FACE = get_face_landmarks(parts_type)

    # Define models.
    # in_channels: C
    #   C: The number of elements in a joint (3: x, y, z).
    # out_channels: The number of words + keywords.
    dictionary = os.path.join(data_dir, "wlasl_100.json")

    _, use_landmarks = get_fullbody_landmarks(
        pose_type=parts_type,
        lhand_type=parts_type,
        rhand_type=parts_type,
        face_type=parts_type)

    use_features = ["x", "y", "z"]
    in_channels = len(use_features) * len(use_landmarks)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    num_words = len(word_dict)
    # Extend dictionary for continuous words recognition.
    word_dict["<pad>"] = {
        "represents": [
            "<pad>"
        ],
        "token": num_words,
        "frequency": -1  # Unknown
    }
    word_dict["<sos>"] = {
        "represents": [
            "<sos>"
        ],
        "token": num_words+1,
        "frequency": -1  # Unknown
    }
    word_dict["<eos>"] = {
        "represents": [
            "<eos>"
        ],
        "token": num_words+2,
        "frequency": -1  # Unknown
    }

    out_channels = len(word_dict)
    max_length_outputs = 10

    model = TransformerConSLR(
        in_channels=in_channels,
        out_channels=out_channels,
        padding_idx=word_dict["<pad>"]["token"],
        start_idx=word_dict["<sos>"]["token"],
        end_idx=word_dict["<eos>"]["token"],
        max_length_outputs=max_length_outputs,
        inter_channels=64,
        num_encoder_head=2,
        num_encoder_layers=2,
        dim_feedforward_encoder=256,
        num_decoder_head=2,
        num_decoder_layers=2,
        dim_feedforward_decoder=256,
        batch_first=True,
        norm_first=True,
        dropout_rate=0.1)

    print(model)

    POSE_ORIGIN = get_calculation_landmarks(
        parts_name="pose", parts_type=parts_type, unit_type="origin")
    POSE_UNIT1 = get_calculation_landmarks(
        parts_name="pose", parts_type=parts_type, unit_type="unit1")
    POSE_UNIT2 = get_calculation_landmarks(
        parts_name="pose", parts_type=parts_type, unit_type="unit2")

    LHAND_ORIGIN = get_calculation_landmarks(
        parts_name="lhand", parts_type=parts_type, unit_type="origin")
    LHAND_UNIT1 = get_calculation_landmarks(
        parts_name="lhand", parts_type=parts_type, unit_type="unit1")
    LHAND_UNIT2 = get_calculation_landmarks(
        parts_name="lhand", parts_type=parts_type, unit_type="unit2")

    RHAND_ORIGIN = get_calculation_landmarks(
        parts_name="rhand", parts_type=parts_type, unit_type="origin")
    RHAND_UNIT1 = get_calculation_landmarks(
        parts_name="rhand", parts_type=parts_type, unit_type="unit1")
    RHAND_UNIT2 = get_calculation_landmarks(
        parts_name="rhand", parts_type=parts_type, unit_type="unit2")

    FACE_ORIGIN = get_calculation_landmarks(
        parts_name="face", parts_type=parts_type, unit_type="origin")
    FACE_UNIT1 = get_calculation_landmarks(
        parts_name="face", parts_type=parts_type, unit_type="unit1")
    FACE_UNIT2 = get_calculation_landmarks(
        parts_name="face", parts_type=parts_type, unit_type="unit2")

    swap_pairs, _ = get_fullbody_swappairs(
        pose_type=parts_type,
        lhand_type=parts_type,
        rhand_type=parts_type,
        face_type=parts_type,)

    # Common process before data augmentation.
    base = [
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero()]
    # Data augmentation.
    daug = [
        RandomHorizontalFlip(
            apply_ratio=0.5,
            num_joints=len(use_landmarks),
            swap_pairs=swap_pairs,
            flip_center=0.5,
            feature_dim=len(use_features),
            include_conf=False),
        RandomAffineTransform2D(
            apply_ratio=0.5,
            # The center of shoulders in pose joints.
            center_joints=POSE_ORIGIN,
            target_joints=np.arange(0, len(use_landmarks)),
            trans_range=[-0.1, 0.1],
            scale_range=[0.8, 1.2],
            rot_range=[-30, 30],
            skew_range=[-5, 5]),
        RandomClip(
            apply_ratio=0.5,
            clip_range=[0.0, 0.2],
            min_apply_size=2),
        RandomTimeWarping(
            apply_ratio=0.5,
            scale_range=[0.5, 1.5],
            min_apply_size=2)]
    # Common process after data augmentation.
    finalize = [
        PartsBasedNormalization(
            pose_head=0,
            pose_num=len(USE_POSE),
            pose_origin=POSE_ORIGIN,
            pose_unit1=POSE_UNIT1,
            pose_unit2=POSE_UNIT2,
            lhand_head=len(USE_POSE),
            lhand_num=len(USE_LHAND),
            lhand_origin=LHAND_ORIGIN,
            lhand_unit1=LHAND_UNIT1,
            lhand_unit2=LHAND_UNIT2,
            rhand_head=len(USE_POSE) + len(USE_LHAND),
            rhand_num=len(USE_RHAND),
            rhand_origin=RHAND_ORIGIN,
            rhand_unit1=RHAND_UNIT1,
            rhand_unit2=RHAND_UNIT2,
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND),
            face_num=len(USE_FACE),
            face_origin=FACE_ORIGIN,
            face_unit1=FACE_UNIT1,
            face_unit2=FACE_UNIT2),
        InsertTokensForS2S(
            sos_token=word_dict["<sos>"]["token"],
            eos_token=word_dict["<eos>"]["token"]),
        ToTensor()]

    train_transforms = Compose(base + daug + finalize)
    base_transforms = Compose(base + finalize)

    track_dir = os.path.join(data_dir, "tracks", "train")
    label_dir = os.path.join(data_dir, "labels", "train")

    dataloader = make_wlasl_dataloader(track_dir, label_dir, dictionary,
                                       batch_size=2, shuffle=True,
                                       transforms=train_transforms)

    # --------------------------------------------------
    # Sanity check.
    # --------------------------------------------------
    # Fetch data.
    sample = next(iter(dataloader))
    tracks = sample["feature"]
    tokens = sample["tokens"]
    feature_pad_mask = sample["feature_pad_mask"]
    tokens_pad_mask = sample["tokens_pad_mask"]
    tokens_causal_mask = make_causal_mask(tokens_pad_mask)
    # `[1, L, L] -> [L, L]`
    tokens_causal_mask = tokens_causal_mask.squeeze(0)

    # Prediction.
    logits = model(feature=tracks,
                   tokens=tokens,
                   feature_causal_mask=None,
                   feature_pad_mask=feature_pad_mask,
                   tokens_causal_mask=tokens_causal_mask,
                   tokens_pad_mask=tokens_pad_mask)

    print(logits.shape)
    probs = torch.softmax(logits, dim=-1)
    print(probs.shape)

    # Predicted words.
    preds = probs.argmax(dim=-1)
    print(preds.shape)
    print(preds)

    # Translate.
    token2word = {value["token"]: key for key, value in word_dict.items()}
    preds = preds.detach().cpu().numpy()
    tokens = tokens.detach().cpu().numpy()

    for i, (pred, token) in enumerate(zip(preds, tokens)):
        pred_words = [token2word[token] for token in pred]
        real_words = [token2word[token] for token in token]
        print(f"{i}th sample.")
        print(pred_words)
        print(real_words)

    loss_fn = CrossEntropyLoss(ignore_index=word_dict["<pad>"]["token"],
                               reduction="mean",
                               label_smoothing=0.1)

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
        transforms=train_transforms)
    val_dataloader = make_wlasl_dataloader(
        os.path.join(data_dir, "tracks", "val"),
        os.path.join(data_dir, "labels", "val"),
        dictionary,
        batch_size=batch_size,
        shuffle=False,
        load_into_ram=load_into_ram,
        transforms=base_transforms)
    test_dataloader = make_wlasl_dataloader(
        os.path.join(data_dir, "tracks", "test"),
        os.path.join(data_dir, "labels", "test"),
        dictionary,
        batch_size=1,
        shuffle=False,
        load_into_ram=load_into_ram,
        transforms=base_transforms)

    model.to(device)

    now = datetime.now()
    save_dir = os.path.join(
        out_dir,
        "A6_1_wlasl_transformer_continuous_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
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
    min_wer = 1000
    min_val = 100000
    min_epoch = 0
    wers = {}
    train_losses = {}
    val_losses = {}
    for t in range(epochs):
        print("-" * 80)
        print(f"Epoch {t+1}")
        train_loss = train_loop_con(
            train_dataloader, model, loss_fn, optimizer, device, use_mask=True)
        val_loss = val_loop_con(val_dataloader, model,
                                loss_fn, device, use_mask=True)
        train_losses[t] = train_loss
        val_losses[t] = val_loss
        # Save checkpoint having minimum validation loss.
        if min_val > val_loss:
            ckpt_path = os.path.join(save_dir, "best_val_loss.ckpt")
            save_checkpoint(ckpt_path, model, t, optimizer)
            min_val = val_loss

        if (t+1) % eval_every_n_epochs == 0:
            wer = test_loop_con(test_dataloader, model, device, use_mask=True,
                                ignore_ids=[
                                    word_dict["<sos>"]["token"],
                                    word_dict["<eos>"]["token"],
                                    word_dict["<pad>"]["token"]
                                ])
            wers[t] = wer
            if wer < min_wer:
                min_wer = wer
                min_epoch = t
                # Save checkpoint having maximum accuracy.
                ckpt_path = os.path.join(save_dir, "best_wer.ckpt")
                save_checkpoint(ckpt_path, model, t, optimizer)

    print(f"The minimum WER is {min_wer} at {min_epoch}th epoch.")

    # Save final checkpoint.
    ckpt_path = os.path.join(save_dir, "last.ckpt")
    save_checkpoint(ckpt_path, model, epochs, optimizer)

    # Save results.
    with open(os.path.join(save_dir, "train_losses.json"), "w") as fwrite:
        json.dump(train_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "val_losses.json"), "w") as fwrite:
        json.dump(val_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "wers.json"), "w") as fwrite:
        json.dump(wers, fwrite, indent=4)

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
