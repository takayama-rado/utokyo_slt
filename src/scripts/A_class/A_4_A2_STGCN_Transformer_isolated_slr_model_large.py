#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_4_A2_STGCN_Transformer_isolated_slr_model_large: Large STGCN-Transformer.
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

from argparse import (
    ArgumentParser,
    Namespace
)
from datetime import datetime

# Third party's modules
import numpy as np

import torch

from torch.nn import CrossEntropyLoss

from torchvision.transforms import Compose

# Local modules
from modules.dataset import (
    make_wlasl_dataloader
)
from modules.defines import (
    get_calculation_landmarks,
    get_pose_landmarks,
    get_lhand_landmarks,
    get_rhand_landmarks,
    get_face_landmarks,
    get_fullbody_landmarks,
    get_parts_connections
)
from modules.graph import GraphSettings
from modules.model import (
    STGCNTransformerIsoSLR
)
from modules.transforms import (
    PartsBasedNormalization,
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    ToTensor)
from modules.train_helpers import (
    train_loop_iso,
    val_loop_iso,
    test_loop_iso,
)
from modules.utils import save_checkpoint


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

    parts_type = "regular"

    USE_POSE = get_pose_landmarks(parts_type)
    USE_LHAND = get_lhand_landmarks(parts_type)
    USE_RHAND = get_rhand_landmarks(parts_type)
    USE_FACE = get_face_landmarks(parts_type)

    # Define models.
    # in_channels: J * C
    #   J: The number of joints (141).
    #   C: The number of elements in a joint (3: x, y, z).
    # out_channels: The number of words + keywords.
    dictionary = os.path.join(data_dir, "wlasl_100.json")

    _, use_landmarks = get_fullbody_landmarks(
        pose_type=parts_type,
        lhand_type=parts_type,
        rhand_type=parts_type,
        face_type=parts_type)

    use_features = ["x", "y", "z"]
    in_channels = len(use_features)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

    # Define graphs.
    _, pose_connections = get_parts_connections(
        use_landmarks, "pose", partstype=parts_type)
    _, lhand_connections = get_parts_connections(
        use_landmarks, "lhand", partstype=parts_type)
    _, rhand_connections = get_parts_connections(
        use_landmarks, "rhand", partstype=parts_type)
    _, face_connections = get_parts_connections(
        use_landmarks, "face", partstype=parts_type)

    connections = np.concatenate([
        pose_connections,
        lhand_connections,
        rhand_connections,
        face_connections], axis=0)

    centers = [
        0,  # Nose
        len(USE_POSE),  # Left wrist
        len(USE_POSE) + len(USE_LHAND),  # Right wrist
        # The center of lip's outline.
        len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND)
    ]

    graph_settings = GraphSettings(
        num_node=len(use_landmarks),
        neighbor_link=connections,  # type: ignore
        centers=centers)  # type: ignore

    model = STGCNTransformerIsoSLR(in_channels,
                                   out_channels,
                                   graph_settings=graph_settings,
                                   num_stgcn_modules=1,
                                   inter_channels=64,
                                   num_head=2,
                                   num_encoder_layers=2,
                                   dim_feedforward=256,
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
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero(),
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
        "A4_A2_wlasl_stgcn_transformer_large_isolated_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
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
