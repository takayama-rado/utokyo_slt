#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_4_1_Transformer_isolated_slr_model: Add Transformer-Encoder layer.
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
import random

from argparse import (
    ArgumentParser,
    Namespace
)
from datetime import datetime

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
from modules.defines import (
    USE_POSE,
    USE_LHAND,
    USE_RHAND,
    USE_FACE,
    get_fullbody_landmarks)
from modules.model import (
    GPoolRecognitionHead
)
from modules.transforms import (
    PartsBasedNormalization,
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    ToTensor)


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
        assert isinstance(self.pose, torch.Tensor)
        feature = feature + self.pose[:, :feature.shape[1]]
        feature = self.dropout(feature)
        return feature


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
            inter_channels=inter_channels)
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
    in_channels = len(use_features) * len(use_landmarks)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

    model = TransformerIsoSLR(in_channels,
                              out_channels,
                              inter_channels=64,
                              num_head=1,
                              num_encoder_layers=1,
                              dim_feedforward=256,
                              batch_first=True,
                              norm_first=True,
                              dropout_rate=0.1)
    print(model)

    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero(),
        PartsBasedNormalization(
            pose_head=0,
            pose_num=len(USE_POSE),
            pose_origin=[11, 12],  # Shoulder
            pose_unit1=[11],
            pose_unit2=[12],
            lhand_head=len(USE_POSE),
            lhand_num=len(USE_LHAND),
            lhand_origin=[0, 2, 5, 9, 13, 17],
            lhand_unit1=[0],
            lhand_unit2=[2, 5, 9, 13, 17],
            rhand_head=len(USE_POSE) + len(USE_LHAND),
            rhand_num=len(USE_RHAND),
            rhand_origin=[0, 2, 5, 9, 13, 17],
            rhand_unit1=[0],
            rhand_unit2=[2, 5, 9, 13, 17],
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND),
            face_num=len(USE_FACE),
            face_origin=[0, 2],
            face_unit1=[7],
            face_unit2=[42]),
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
        "A4_1_wlasl_transformer_isolated_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
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
