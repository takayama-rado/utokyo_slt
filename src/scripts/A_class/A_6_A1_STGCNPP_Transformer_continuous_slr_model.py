#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_6_A1_STGCNPP_Transformer_continuous_slr_model: STGCNPP-Transformer CSLR model.
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
from modules.defines import (
    get_calculation_landmarks,
    get_pose_landmarks,
    get_lhand_landmarks,
    get_rhand_landmarks,
    get_face_landmarks,
    get_fullbody_landmarks,
    get_fullbody_swappairs,
    get_parts_connections
)
from modules.graph import GraphSettings
from modules.transformer_helper import (
    make_causal_mask
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
from modules.model import (
    TransformerConBase
)
from modules.stgcn import (
    STGCNPPModule
)
from modules.transforms import (
    InsertTokensForS2S,
    RandomHorizontalFlip,
    RandomAffineTransform2D,
    RandomClip,
    RandomTimeWarping,
    PartsBasedNormalization,
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    ToTensor)
from modules.train_helpers import (
    train_loop_con,
    val_loop_con,
    test_loop_con
)
from modules.utils import save_checkpoint


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

    def _build_feature_extractor(self,  # type: ignore
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

    model = STGCNPPTransformerConSLR(
        in_channels=in_channels,
        out_channels=out_channels,
        graph_settings=graph_settings,
        num_stgcn_modules=2,
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
        "A6_A1_wlasl_stgcnpp_transformer_continuous_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
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
