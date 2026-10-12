#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_4_B1_Check_Transformer_attention: Check attention outputs.
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
)
from modules.model import (
    TransformerIsoSLR
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
        ckpt_path = args.ckpt_path
        target_data_index = args.target_data_index
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    # Load hyper parameter.
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "hparams.json"), "r") as fread:
        hparams_orig = json.load(fread)
    print(hparams_orig)

    # Create model.
    in_channels = hparams_orig["in_channels"]
    out_channels = hparams_orig["out_channels"]

    model = TransformerIsoSLR(in_channels,
                              out_channels,
                              inter_channels=64,
                              num_head=2,
                              num_encoder_layers=2,
                              dim_feedforward=256,
                              batch_first=True,
                              norm_first=True,
                              dropout_rate=0.1)
    print(model)

    # Define data loader.
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
    in_channels = len(use_features) * len(use_landmarks)
    with open(dictionary, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

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
                                       batch_size=1, shuffle=True,
                                       transforms=transforms)

    # Translate.
    token2word = {value["token"]: key for key, value in word_dict.items()}
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using {device} for computation.")
    model.to(device)

    # Fetch data.
    model.eval()
    pred_words = None
    real_words = None
    i = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            tracks = sample["feature"]
            tokens = sample["tokens"]
            mask = sample["feature_pad_mask"]

            tracks = tracks.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Prediction.
            logits = model(tracks, feature_pad_mask=mask)
            probs = torch.softmax(logits, dim=1)

            # Predicted words.
            preds = probs.argmax(dim=1)

            # Translate.
            preds = preds.detach().cpu().numpy().flatten()
            tokens = tokens.detach().cpu().numpy().flatten()

            pred_words = [token2word[token] for token in preds]
            real_words = [token2word[token] for token in tokens]
            if target_data_index >= 0 and target_data_index == i:
                break
            if target_data_index < 0 and pred_words[0] == real_words[0]:
                break

    print(f"Data number: {i}")
    print(pred_words)
    print(real_words)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--target_data_index", type=int, default=-1)

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
