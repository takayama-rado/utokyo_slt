#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""train_helpers: Helper functions for training.
-------------------------------------------------------------------------------


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

from Levenshtein import distance

# Local modules
from modules.transformer_helper import make_causal_mask

# Execution settings
VERSION = "%(prog)s dev"


def train_loop_iso(dataloader,
                   model,
                   loss_fn,
                   optimizer,
                   device,
                   use_mask=False):
    size = len(dataloader.dataset)

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

        # Print current loss per 100 steps.
        if batch_idx % 10 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")


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


def train_loop_con(dataloader,
                   model,
                   loss_fn,
                   optimizer,
                   device,
                   use_mask=False):
    size = len(dataloader.dataset)
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

        # Print current loss per 100 steps.
        if batch_idx % 10 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")


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


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
