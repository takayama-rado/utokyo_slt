#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_2_2_Temporal_CNN_isolated_slr_model: Add CNN layer.
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


class FExtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)

        # Keep the same size.
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(out_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, feature):
        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        feature = self.linear(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.norm1(feature)
        feature = self.activation(feature)

        feature = self.conv(feature)
        feature = self.norm2(feature)
        feature = self.activation(feature)
        return feature


class CNNIsoSLR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.fext = FExtBlock(in_channels, 64, kernel_size=kernel_size)
        self.head = GPoolRecognitionHead(64, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        """Forward computation.

        # Args:
          - inputs: `[N, in_channels, T, J]`
          - pad_mask: `[N, T]`
        # Returns:
          - logits: `[N, out_channels]`
        """
        # `[N, C, T, J] -> [N, T, J, C] -> [N, T, J*C] -> [N, J*C, T]`
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = feature.permute([0, 2, 1])

        feature = self.fext(feature)
        logits = self.head(feature, feature_pad_mask)
        return logits


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
    in_channels = len(use_landmarks) * len(use_features)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

    kernel_size = 9
    model = CNNIsoSLR(in_channels, out_channels, kernel_size=kernel_size)
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
        "A2_2_wlasl_cnn_isolated_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
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
