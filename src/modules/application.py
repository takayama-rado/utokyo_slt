#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""utils: Utility functions.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from dataclasses import (
    dataclass,
    field)
from typing import (
    List
)

# Third party's modules
import numpy as np

import torch
from torch.nn import CrossEntropyLoss

from torchvision.transforms import Compose


# Local modules
from . import transforms as tr

from .dataset import (
    make_wlasl_dataloader)
from .defines import (
    get_calculation_landmarks,
    get_fullbody_landmarks,
    get_fullbody_swappairs,
    get_parts_connections,
    get_pose_landmarks,
    get_lhand_landmarks,
    get_rhand_landmarks,
    get_face_landmarks,
)
from .graph import (
    GraphSettings)
from .model import (
    SimpleIsoSLR,
    STGCNIsoSLR,
    STGCNPPIsoSLR,
    STGCNWMIsoSLR,
    STGCNPPWMIsoSLR,
    TransformerIsoSLR,
    STGCNTransformerIsoSLR,
    STGCNPPTransformerIsoSLR,
    STGCNWMTransformerIsoSLR,
    STGCNPPWMTransformerIsoSLR,
    TransformerConSLR,
    STGCNTransformerConSLR,
    STGCNPPTransformerConSLR,
    STGCNWMTransformerConSLR,
    STGCNPPWMTransformerConSLR)
from .train_helpers import (
    train_loop_con,
    val_loop_con,
    test_loop_con,
    train_loop_iso,
    val_loop_iso,
    test_loop_iso
)
from .utils import (
    load_checkpoint,
    save_checkpoint)


# Execution settings
VERSION = "%(prog)s dev"


def run_islr_training(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        use_tmask,
        epochs,
        eval_every_n_epochs,
        save_dir,
        ckpt_path=None):
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print("Load checkpoint...")
        model, start_epoch, optimizer, _ = load_checkpoint(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer)
        print("Done.")
    else:
        start_epoch = 0

    print("Start training...")
    max_acc = 0
    min_val = 100000
    max_epoch = 0
    accuracies = {}
    val_losses = {}
    train_losses = {}
    for t in range(start_epoch, epochs):
        print("-" * 80)
        print(f"Epoch {t+1}")
        train_loss = train_loop_iso(train_dataloader, model, loss_fn,
                                    optimizer, device, use_mask=use_tmask)
        val_loss = val_loop_iso(val_dataloader, model, loss_fn,
                                device, use_mask=use_tmask)
        train_losses[t] = train_loss
        val_losses[t] = val_loss
        if min_val > val_loss:
            ckpt_path = os.path.join(save_dir, "best_val_loss.ckpt")
            save_checkpoint(ckpt_path, model, t, optimizer)
        if (t+1) % eval_every_n_epochs == 0:
            acc = test_loop_iso(test_dataloader, model,
                                device, use_mask=use_tmask)
            accuracies[t] = acc
            if acc > max_acc:
                max_acc = acc
                max_epoch = t
                ckpt_path = os.path.join(save_dir, "best_acc.ckpt")
                save_checkpoint(ckpt_path, model, t, optimizer)
    print(f"The maximum accuracy is {max_acc} at {max_epoch}th epoch.")

    # Save last model.
    ckpt_path = os.path.join(save_dir, "last.ckpt")
    save_checkpoint(ckpt_path, model, t, optimizer)
    with open(os.path.join(save_dir, "train_losses.json"), "w") as fwrite:
        json.dump(train_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "val_losses.json"), "w") as fwrite:
        json.dump(val_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "accuracies.json"), "w") as fwrite:
        json.dump(accuracies, fwrite, indent=4)
    return accuracies


def run_cslr_training(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        word_dict,
        loss_fn,
        optimizer,
        device,
        use_tmask,
        epochs,
        eval_every_n_epochs,
        save_dir,
        ckpt_path=None):
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print("Load checkpoint...")
        model, start_epoch, optimizer, _ = load_checkpoint(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer)
        print("Done.")
    else:
        start_epoch = 0

    print("Start training...")
    min_awer = 100
    min_val = 100000
    min_epoch = 0
    word_error_rates = {}
    train_losses = {}
    val_losses = {}
    for t in range(start_epoch, epochs):
        print("-" * 80)
        print(f"Epoch {t+1}")
        train_loss = train_loop_con(train_dataloader, model, loss_fn,
                                    optimizer, device, use_tmask)
        val_loss = val_loop_con(val_dataloader, model, loss_fn,
                                device, use_tmask)
        train_losses[t] = train_loss
        val_losses[t] = val_loss
        if min_val > val_loss:
            ckpt_path = os.path.join(save_dir, "best_val_loss.ckpt")
            save_checkpoint(ckpt_path, model, t, optimizer)
        if (t+1) % eval_every_n_epochs == 0:
            awer = test_loop_con(test_dataloader, model,
                                 device, use_tmask,
                                 ignore_ids=[
                                     word_dict["<pad>"]["token"],
                                     word_dict["<sos>"]["token"],
                                     word_dict["<eos>"]["token"]
                                 ])
            word_error_rates[t] = awer
            if awer < min_awer:
                min_awer = awer
                min_epoch = t
                ckpt_path = os.path.join(save_dir, "best_wer.ckpt")
                save_checkpoint(ckpt_path, model, t, optimizer)
    print(f"The minimum WER is {min_awer} at {min_epoch}th epoch.")

    # Save last model.
    ckpt_path = os.path.join(save_dir, "last.ckpt")
    save_checkpoint(ckpt_path, model, t, optimizer)
    with open(os.path.join(save_dir, "train_losses.json"), "w") as fwrite:
        json.dump(train_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "val_losses.json"), "w") as fwrite:
        json.dump(val_losses, fwrite, indent=4)
    with open(os.path.join(save_dir, "wer.json"), "w") as fwrite:
        json.dump(word_error_rates, fwrite, indent=4)
    return word_error_rates


@dataclass
class AppSettings:
    # Basic settings.
    track_rootdir: str
    label_rootdir: str
    label_dictionary: str
    model_type: str = "stgcnpp_transformer"
    task_type: str = "continuous"
    inter_channels: int = 64
    use_features: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    # Training settings.
    batch_size: int = 32
    load_into_ram: bool = False
    epochs: int = 3
    learning_rate: float = 3e-4
    num_workers: int = 0
    eval_every_n_epochs: int = 1
    # Training Options.
    use_tmask: bool = True
    use_compact_parts: bool = True
    remove_face: bool = False
    use_daug: bool = True
    label_smoothing: float = 0.1
    # STGCN settings.
    num_stgcn_modules: int = 1
    # Transformer encoder settings.
    num_encoder_head: int = 2
    num_encoder_layers: int = 2
    dim_feedforward_encoder: int = 256
    # Transformer decoder settings.
    num_decoder_head: int = 2
    num_decoder_layers: int = 2
    dim_feedforward_decoder: int = 256
    max_length_outputs: int = 20

    def __post_init__(self):
        # Check arguments.
        assert self.task_type in ["isolated", "continuous"]
        assert self.model_type in [
            "simple", "stgcn", "stgcnpp", "masked_stgcn", "masked_stgcnpp",
            "transformer", "stgcn_transformer", "stgcnpp_transformer",
            "masked_stgcn_transformer", "masked_stgcnpp_transformer"]
        if self.task_type == "continuous" and self.model_type in [
                "simple", "stgcn", "stgcnpp",
                "masked_stgcn", "masked_stgcnpp"]:
            raise ValueError(f"The model_type:{self.model_type} is only for"
                             "isolated SLR task.")

        # Load dictionary.
        with open(self.label_dictionary, "r") as fread:
            word_dict = json.load(fread)
        if self.task_type == "continuous":
            num_words = len(word_dict)
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
        self.word_dict = word_dict

        # Get landmark definition.
        self.pose_type = "default" if not self.use_compact_parts else "compact"
        self.lhand_type = "default" if not self.use_compact_parts \
            else "compact"
        self.rhand_type = "default" if not self.use_compact_parts \
            else "compact"
        if self.remove_face:
            self.face_type = "unused"
        else:
            self.face_type = "default" if not self.use_compact_parts \
                else "compact"
        _, self.use_landmarks = get_fullbody_landmarks(
            pose_type=self.pose_type,
            lhand_type=self.lhand_type,
            rhand_type=self.rhand_type,
            face_type=self.face_type)
        self.swap_pairs, _ = get_fullbody_swappairs(
            pose_type=self.pose_type,
            lhand_type=self.lhand_type,
            rhand_type=self.rhand_type,
            face_type=self.face_type)

    def get_transforms(self):
        # Define transformation.
        base = [
            tr.SelectLandmarksAndFeature(
                self.use_landmarks, self.use_features),
            tr.ReplaceNan(),
            tr.FilterZero()]
        daug = [
            tr.RandomHorizontalFlip(
                apply_ratio=0.5,
                num_joints=len(self.use_landmarks),
                swap_pairs=self.swap_pairs,
                flip_center=0.5,
                feature_dim=len(self.use_features),
                include_conf=False),
            tr.RandomAffineTransform2D(
                apply_ratio=0.5,
                center_joints=[0],  # Nose
                target_joints=np.arange(0, len(self.use_landmarks)),
                trans_range=[-0.1, 0.1],
                scale_range=[0.8, 1.2],
                rot_range=[-30, 30],
                skew_range=[-5, 5]
            ),
            tr.RandomClip(
                apply_ratio=0.5,
                clip_range=[0.0, 0.5],
                min_apply_size=2
            ),
            tr.RandomTimeWarping(
                apply_ratio=0.5,
                scale_range=[0.5, 1.5],
                min_apply_size=2
            )
        ]

        # Pose parameters.
        pose_head = 0
        pose_num = len(get_pose_landmarks(self.pose_type))
        pose_origin = get_calculation_landmarks("pose", self.pose_type,
                                                "origin")
        pose_unit1 = get_calculation_landmarks("pose", self.pose_type,
                                               "unit1")
        pose_unit2 = get_calculation_landmarks("pose", self.pose_type,
                                               "unit2")
        # Lhand parameters.
        lhand_head = pose_num
        lhand_num = len(get_lhand_landmarks(self.lhand_type))
        lhand_origin = get_calculation_landmarks("lhand", self.lhand_type,
                                                 "origin")
        lhand_unit1 = get_calculation_landmarks("lhand", self.lhand_type,
                                                "unit1")
        lhand_unit2 = get_calculation_landmarks("lhand", self.lhand_type,
                                                "unit2")
        # Rhand parameters.
        rhand_head = pose_num + lhand_num
        rhand_num = len(get_rhand_landmarks(self.rhand_type))
        rhand_origin = get_calculation_landmarks("rhand", self.rhand_type,
                                                 "origin")
        rhand_unit1 = get_calculation_landmarks("rhand", self.rhand_type,
                                                "unit1")
        rhand_unit2 = get_calculation_landmarks("rhand", self.rhand_type,
                                                "unit2")
        # Face parameters.
        if self.remove_face:
            face_num = 0
        else:
            face_num = len(get_face_landmarks(self.face_type))
        face_head = pose_num + lhand_num + rhand_num
        face_origin = get_calculation_landmarks("face", self.face_type,
                                                "origin")
        face_unit1 = get_calculation_landmarks("face", self.face_type,
                                               "unit1")
        face_unit2 = get_calculation_landmarks("face", self.face_type,
                                               "unit2")

        finalize = [
            tr.PartsBasedNormalization(
                pose_head=pose_head,
                pose_num=pose_num,
                pose_origin=pose_origin,
                pose_unit1=pose_unit1,
                pose_unit2=pose_unit2,
                lhand_head=lhand_head,
                lhand_num=lhand_num,
                lhand_origin=lhand_origin,
                lhand_unit1=lhand_unit1,
                lhand_unit2=lhand_unit2,
                rhand_head=rhand_head,
                rhand_num=rhand_num,
                rhand_origin=rhand_origin,
                rhand_unit1=rhand_unit1,
                rhand_unit2=rhand_unit2,
                face_head=face_head,
                face_num=face_num,
                face_origin=face_origin,
                face_unit1=face_unit1,
                face_unit2=face_unit2
            )
        ]
        if self.task_type == "continuous":
            finalize.append(tr.InsertTokensForS2S(
                sos_token=self.word_dict["<sos>"]["token"],
                eos_token=self.word_dict["<eos>"]["token"]))
        finalize.append(tr.ToTensor())
        if self.use_daug:
            train_transforms = Compose(base + daug + finalize)
        else:
            train_transforms = Compose(base + finalize)
        base_transforms = Compose(base + finalize)
        return train_transforms, base_transforms

    def get_dataloaders(self):
        train_transforms, base_transforms = self.get_transforms()

        print("Start data loading...")
        if self.load_into_ram is True:
            print("Data pre-loading is applied.")
        if self.task_type == "isolated":
            token_padding_val = -1
        else:
            token_padding_val = self.word_dict["<pad>"]["token"]

        train_dataloader = make_wlasl_dataloader(
            track_dir=os.path.join(self.track_rootdir, "train"),
            label_dir=os.path.join(self.label_rootdir, "train"),
            label_dictionary=self.label_dictionary,
            batch_size=self.batch_size,
            load_into_ram=self.load_into_ram,
            shuffle=True,
            convert_to_channel_first=True,
            token_padding_val=token_padding_val,
            num_workers=self.num_workers,
            transforms=train_transforms)
        val_dataloader = make_wlasl_dataloader(
            track_dir=os.path.join(self.track_rootdir, "val"),
            label_dir=os.path.join(self.label_rootdir, "val"),
            label_dictionary=self.label_dictionary,
            batch_size=self.batch_size,
            load_into_ram=self.load_into_ram,
            shuffle=False,
            convert_to_channel_first=True,
            token_padding_val=token_padding_val,
            transforms=base_transforms)
        test_dataloader = make_wlasl_dataloader(
            track_dir=os.path.join(self.track_rootdir, "test"),
            label_dir=os.path.join(self.label_rootdir, "test"),
            label_dictionary=self.label_dictionary,
            batch_size=1,
            load_into_ram=self.load_into_ram,
            shuffle=False,
            convert_to_channel_first=True,
            token_padding_val=token_padding_val,
            transforms=base_transforms)
        print("Done.")
        return train_dataloader, val_dataloader, test_dataloader

    def get_graph_settings(self):
        partstype = "default" if not self.use_compact_parts else "compact"
        _, p_connections = get_parts_connections(
            self.use_landmarks, "pose", partstype)
        _, l_connections = get_parts_connections(
            self.use_landmarks, "lhand", partstype)
        _, r_connections = get_parts_connections(
            self.use_landmarks, "rhand", partstype)
        if not self.remove_face:
            _, f_connections = get_parts_connections(self.use_landmarks,
                                                     "face",
                                                     partstype)
            connections = np.concatenate([p_connections, l_connections,
                                          r_connections, f_connections],
                                         axis=0)
        else:
            connections = np.concatenate([p_connections, l_connections,
                                          r_connections], axis=0)
        # Centers should be the first landmarks of each body-part.
        pose_num = len(get_pose_landmarks(partstype))
        lhand_num = len(get_lhand_landmarks(partstype))
        rhand_num = len(get_rhand_landmarks(partstype))
        centers = [
            0,  # Nose
            pose_num,  # LWrist
            pose_num + lhand_num,  # RWrist
            pose_num + lhand_num + rhand_num  # The center of lip's outline.
        ]

        graph_settings = GraphSettings(
            num_node=len(self.use_landmarks),
            neighbor_link=connections,
            centers=centers)
        return graph_settings

    def _create_simple_islr(self, in_channels, out_channels):
        model = SimpleIsoSLR(in_channels=in_channels,
                             out_channels=out_channels)
        return model

    def _create_stgcn_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNIsoSLR(graph_settings=graph_settings,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            inter_channels=self.inter_channels,
                            num_stgcn_modules=self.num_stgcn_modules)
        return model

    def _create_stgcnpp_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPIsoSLR(graph_settings=graph_settings,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              inter_channels=self.inter_channels,
                              num_stgcn_modules=self.num_stgcn_modules)
        return model

    def _create_masked_stgcn_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNWMIsoSLR(graph_settings=graph_settings,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              inter_channels=self.inter_channels,
                              num_stgcn_modules=self.num_stgcn_modules)
        return model

    def _create_masked_stgcnpp_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPWMIsoSLR(graph_settings=graph_settings,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                inter_channels=self.inter_channels,
                                num_stgcn_modules=self.num_stgcn_modules)
        return model

    def _create_transformer_islr(self, in_channels, out_channels):
        model = TransformerIsoSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            inter_channels=self.inter_channels,
            num_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward_encoder)
        return model

    def _create_stgcn_transformer_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNTransformerIsoSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            inter_channels=self.inter_channels,
            num_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward_encoder)
        return model

    def _create_stgcnpp_transformer_islr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPTransformerIsoSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            inter_channels=self.inter_channels,
            num_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward_encoder)
        return model

    def _create_masked_stgcn_transformer_islr(self,
                                              in_channels,
                                              out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNWMTransformerIsoSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            inter_channels=self.inter_channels,
            num_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward_encoder)
        return model

    def _create_masked_stgcnpp_transformer_islr(self,
                                                in_channels,
                                                out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPWMTransformerIsoSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            inter_channels=self.inter_channels,
            num_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward_encoder)
        return model

    def _create_transformer_cslr(self, in_channels, out_channels):
        model = TransformerConSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            padding_idx=self.word_dict["<pad>"]["token"],
            start_idx=self.word_dict["<sos>"]["token"],
            end_idx=self.word_dict["<eos>"]["token"],
            max_length_outputs=self.max_length_outputs,
            inter_channels=self.inter_channels,
            num_encoder_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward_encoder=self.dim_feedforward_encoder,
            num_decoder_head=self.num_decoder_head,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward_decoder=self.dim_feedforward_decoder,
            batch_first=True,
            norm_first=True,
            dropout_rate=0.1)
        return model

    def _create_stgcn_transformer_cslr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNTransformerConSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            padding_idx=self.word_dict["<pad>"]["token"],
            start_idx=self.word_dict["<sos>"]["token"],
            end_idx=self.word_dict["<eos>"]["token"],
            max_length_outputs=self.max_length_outputs,
            inter_channels=self.inter_channels,
            num_encoder_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward_encoder=self.dim_feedforward_encoder,
            num_decoder_head=self.num_decoder_head,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward_decoder=self.dim_feedforward_decoder,
            batch_first=True,
            norm_first=True,
            dropout_rate=0.1)
        return model

    def _create_stgcnpp_transformer_cslr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPTransformerConSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            padding_idx=self.word_dict["<pad>"]["token"],
            start_idx=self.word_dict["<sos>"]["token"],
            end_idx=self.word_dict["<eos>"]["token"],
            max_length_outputs=self.max_length_outputs,
            inter_channels=self.inter_channels,
            num_encoder_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward_encoder=self.dim_feedforward_encoder,
            num_decoder_head=self.num_decoder_head,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward_decoder=self.dim_feedforward_decoder,
            batch_first=True,
            norm_first=True,
            dropout_rate=0.1)
        return model

    def _create_masked_stgcn_transformer_cslr(self, in_channels, out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNWMTransformerConSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            padding_idx=self.word_dict["<pad>"]["token"],
            start_idx=self.word_dict["<sos>"]["token"],
            end_idx=self.word_dict["<eos>"]["token"],
            max_length_outputs=self.max_length_outputs,
            inter_channels=self.inter_channels,
            num_encoder_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward_encoder=self.dim_feedforward_encoder,
            num_decoder_head=self.num_decoder_head,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward_decoder=self.dim_feedforward_decoder,
            batch_first=True,
            norm_first=True,
            dropout_rate=0.1)
        return model

    def _create_masked_stgcnpp_transformer_cslr(self,
                                                in_channels,
                                                out_channels):
        graph_settings = self.get_graph_settings()
        model = STGCNPPWMTransformerConSLR(
            in_channels=in_channels,
            out_channels=out_channels,
            graph_settings=graph_settings,
            num_stgcn_modules=self.num_stgcn_modules,
            padding_idx=self.word_dict["<pad>"]["token"],
            start_idx=self.word_dict["<sos>"]["token"],
            end_idx=self.word_dict["<eos>"]["token"],
            max_length_outputs=self.max_length_outputs,
            inter_channels=self.inter_channels,
            num_encoder_head=self.num_encoder_head,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward_encoder=self.dim_feedforward_encoder,
            num_decoder_head=self.num_decoder_head,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward_decoder=self.dim_feedforward_decoder,
            batch_first=True,
            norm_first=True,
            dropout_rate=0.1)
        return model

    def create_model(self):
        if self.model_type in ["simple", "transformer"]:
            in_channels = len(self.use_landmarks) * len(self.use_features)
        else:
            in_channels = len(self.use_features)
        out_channels = len(self.word_dict)

        if self.task_type == "isolated":
            func_map = {
                "simple": self._create_simple_islr,
                "stgcn": self._create_stgcn_islr,
                "stgcnpp": self._create_stgcnpp_islr,
                "masked_stgcn": self._create_masked_stgcn_islr,
                "masked_stgcnpp": self._create_masked_stgcnpp_islr,
                "transformer": self._create_transformer_islr,
                "stgcn_transformer": self._create_stgcn_transformer_islr,
                "stgcnpp_transformer": self._create_stgcnpp_transformer_islr,
                "masked_stgcn_transformer":
                    self._create_masked_stgcn_transformer_islr,
                "masked_stgcnpp_transformer":
                    self._create_masked_stgcnpp_transformer_islr,
            }
            model = func_map[self.model_type](in_channels, out_channels)
        elif self.task_type == "continuous":
            func_map = {
                "transformer": self._create_transformer_cslr,
                "stgcn_transformer": self._create_stgcn_transformer_cslr,
                "stgcnpp_transformer": self._create_stgcnpp_transformer_cslr,
                "masked_stgcn_transformer":
                    self._create_masked_stgcn_transformer_cslr,
                "masked_stgcnpp_transformer":
                    self._create_masked_stgcnpp_transformer_cslr,
            }
            model = func_map[self.model_type](in_channels, out_channels)
        return model

    def get_loss_func(self):
        ignore_index = -100 if self.task_type == "isolated" else \
            self.word_dict["<pad>"]["token"]
        loss_fn = CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean",
            label_smoothing=self.label_smoothing)
        return loss_fn

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return optimizer


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
