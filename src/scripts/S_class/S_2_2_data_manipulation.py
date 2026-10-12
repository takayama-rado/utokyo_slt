#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_2_2_data_manipulation: Data manipulation.
-------------------------------------------------------------------------------

This script is licensed under Apache License 2.0
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

# Third party's modules
import cv2

import matplotlib

import numpy as np

import torch

from matplotlib import pyplot as plt

from torch.utils.data import (
    Dataset,
    DataLoader)
from torchvision.transforms import Compose

# Local modules


matplotlib.use("Agg")

# 描画する頂点を定義
# 上半身だけ
USE_POSE = np.arange(0, 23)

# 左手，右手は全部使用
# 最終的に，身体
USE_LHAND = np.arange(33, 33+21)
USE_RHAND = np.arange(33+21, 33+21+21)

# 顔は唇，鼻，両目を使用
USE_LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
USE_NOSE = [98, 1, 327, 2]
USE_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173]
USE_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398]
USE_FACE = np.array(USE_LIP + USE_NOSE + USE_REYE + USE_LEYE) + 33 + 21 + 21

# 線で繋ぐ頂点の組を定義
# 上半身
POSE_CONNECTIONS = np.array([
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
    [5, 6], [6, 8], [9, 10], [11, 12], [11, 13],
    [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
    [18, 20]])
# 左手，右手
HAND_CONNECTIONS = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]])
LHAND_CONNECTIONS = HAND_CONNECTIONS + 33
RHAND_CONNECTIONS = HAND_CONNECTIONS + 33 + 21
# 顔
LIP_CONNECTIONS = np.array([
    [61, 185], [185, 40], [40, 39], [39, 37], [37, 0],
    [0, 267], [267, 269], [269, 270], [270, 409], [409, 291],
    [291, 375], [375, 321], [321, 405], [405, 314], [314, 17],
    [17, 84], [84, 181], [181, 91], [91, 146], [146, 61],
    [78, 191], [191, 80], [80, 81], [81, 82], [82, 13],
    [13, 312], [312, 311], [311, 310], [310, 415], [415, 308],
    [308, 324], [324, 318], [318, 402], [402, 317], [317, 14],
    [14, 87], [87, 178], [178, 88], [88, 95], [95, 78]])
NOSE_CONNECTIONS = np.array([
    [98, 1], [1, 327], [327, 2], [2, 98]
])
REYE_CONNECTIONS = np.array([
    [33, 7], [7, 163], [163, 144], [144, 145], [145, 153],
    [153, 154], [154, 155], [155, 133], [133, 173], [173, 157],
    [157, 158], [158, 159], [159, 160], [160, 161], [161, 246],
    [246, 33]
])
LEYE_CONNECTIONS = np.array([
    [263, 249], [249, 390], [390, 373], [373, 374], [374, 380],
    [380, 381], [381, 382], [382, 362], [362, 398], [398, 384],
    [384, 385], [385, 386], [386, 387], [387, 388], [388, 466],
    [466, 263]])
FACE_CONNECTIONS = np.concatenate([LIP_CONNECTIONS,
                                   NOSE_CONNECTIONS,
                                   REYE_CONNECTIONS,
                                   LEYE_CONNECTIONS], axis=0) + 33 + 21 + 21


class WlaslDataset(Dataset):
    def __init__(self, track_dir, label_dir, label_dictionary,
                 convert_to_channel_first=False,
                 load_into_ram=False, transforms=None):

        # Load track file pointers.
        track_dir = os.path.abspath(track_dir)
        files = os.listdir(track_dir)
        files = [fin for fin in files if fin.split(".")[-1] == "npy"]
        assert len(files) > 0, f"Can not find npy files in {track_dir}."
        self.track_files = sorted(
            [os.path.abspath(os.path.join(track_dir, fin)) for fin in files])

        # Load label file pointers.
        label_dir = os.path.abspath(label_dir)
        files = os.listdir(label_dir)
        files = [fin for fin in files if fin.split(".")[-1] == "json"]
        assert len(files) > 0, f"Can not find json files in {label_dir}."
        self.label_files = sorted(
            [os.path.abspath(os.path.join(label_dir, fin)) for fin in files])

        # Check pairs.
        message = f"{len(self.track_files)}:{len(self.label_files)}"
        assert len(self.track_files) == len(self.label_files), message
        for track, label in zip(self.track_files, self.label_files):
            base_track = os.path.basename(track).split(".")[0]
            base_label = os.path.basename(label).split(".")[0]
            message = f"Invalid file pairs, {track}, {label}"
            assert base_track == base_label, message

        if load_into_ram is True:
            self.track_files = [np.load(fin) for fin in self.track_files]
            temp = []
            for fin in self.label_files:
                with open(fin, "r") as fread:
                    temp.append(json.load(fread))
            self.label_files = temp

        # Load dictionary.
        with open(label_dictionary, "r") as fread:
            self.dictionary = json.load(fread)

        self.convert_to_channel_first = convert_to_channel_first
        self.transforms = transforms

    def __len__(self):
        return len(self.track_files)

    def __getitem__(self, idx):
        if isinstance(self.track_files[idx], str):
            # Load tracking data.
            track = np.load(self.track_files[idx])
        else:
            track = self.track_files[idx]

        # Landmarks tracked by Holistic should include a single person.
        track = track[0]
        if self.convert_to_channel_first:
            # `[T, J, C] -> [C, T, J]` for consistency.
            track = np.transpose(track, [2, 0, 1])

        if isinstance(self.label_files[idx], str):
            # Load label and convert it to tokens.
            with open(self.label_files[idx], "r") as fread:
                annot = json.load(fread)
        else:
            annot = self.label_files[idx]
        items = annot["items"]
        # Extract contents.
        contents = [item["content"] for item in items]
        # Convert contents to tokens.
        tokens = [self.dictionary[cnt]["token"] for cnt in contents]
        tokens = np.array(tokens, dtype=np.int64)

        sample = {"feature": track, "tokens": tokens}
        if self.transforms is not None:
            sample = self.transforms(sample)
        else:
            new_sample = {}
            for key, val in sample.items():
                if key == "feature":
                    assert len(val.shape) == 3, f"{val.shape}"
                if val.dtype == np.float64:
                    val = val.astype(np.float32)
                val = torch.from_numpy(val)
                new_sample[key] = val
            sample = new_sample
        return sample


def merge(sequences, merged_shape, padding_val=0):
    merged = torch.full(tuple(merged_shape),
                        padding_val,
                        dtype=sequences[0].dtype)
    if len(merged_shape) == 2:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0]] = seq
    elif len(merged_shape) == 3:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1]] = seq
    elif len(merged_shape) == 4:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1], :seq.shape[2]] = seq
    elif len(merged_shape) == 5:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1],
                   :seq.shape[2], :seq.shape[3]] = seq
    return merged


def merge_padded_batch(batch,
                       max_length_inputs=None,
                       max_length_outputs=None,
                       token_padding_val=-1):
    feature_batch = [sample["feature"] for sample in batch]
    tokens_batch = [sample["tokens"] for sample in batch]

    merged_shape = None
    try:
        # `[N, C, T, J]`
        num_channels = feature_batch[0].shape[0]
        num_kpts = feature_batch[0].shape[-1]
        if max_length_inputs is None:
            max_length_inputs = max([feature.shape[1]
                                     for feature in feature_batch])
        merged_shape = [len(batch), num_channels, max_length_inputs,
                        num_kpts]
        merged_feature = merge(feature_batch, merged_shape,
                               padding_val=0)

        if max_length_outputs is None:
            max_length_outputs = max([tokens.shape[0]
                                      for tokens in tokens_batch])
        merged_shape = [len(batch), max_length_outputs]
        merged_tokens = merge(tokens_batch, merged_shape,
                              padding_val=token_padding_val)

    except Exception as inst:
        print(inst)
        with open("error_dump.txt", "w") as fwrite:
            message = str(inst) + "\n" \
                + str(merged_shape) + "\n" \
                + str(feature_batch) + "\n" \
                + str(tokens_batch) + "\n"
            fwrite.write(message)
        raise inst

    # Assign True if all channels are 0.
    # We first check channel axis (dim=1), and then check kpts axis (dim=-1).
    feature_pad_mask = (merged_feature == 0).all(dim=1).all(dim=-1)
    tokens_pad_mask = merged_tokens == token_padding_val
    # 0: Pad, 1: Signal.
    feature_pad_mask = torch.logical_not(feature_pad_mask)
    tokens_pad_mask = torch.logical_not(tokens_pad_mask)
    retval = {
        "feature": merged_feature,
        "tokens": merged_tokens,
        "feature_pad_mask": feature_pad_mask,
        "tokens_pad_mask": tokens_pad_mask
    }
    return retval


class ReplaceNan():
    """ Replace NaN value in the feature.
    """

    def __init__(self, replace_val=0.0) -> None:
        self.replace_val = replace_val

    def __call__(self,
                 data):
        feature = data["feature"]
        feature[np.isnan(feature)] = self.replace_val
        data["feature"] = feature
        return data


class FilterZero():
    """ Filter Zero value in the feature.
    """

    def __init__(self,
                 apply_ratio=1.0) -> None:
        self.apply_ratio = apply_ratio

    def __call__(self,
                 data):
        if random.random() > self.apply_ratio:
            return data
        feature = data["feature"]
        # Drop.
        zmask = feature == 0.0
        zmask = np.all(zmask, axis=(0, 2))
        zmask = np.logical_not(zmask)
        feature = feature[:, zmask, :]
        data["feature"] = feature
        return data


class SelectLandmarksAndFeature():
    """ Select joint and feature.
    """

    def __init__(self, landmarks, features=["x", "y", "z", "c"]):
        self.landmarks = landmarks
        _features = []
        if "x" in features:
            _features.append(0)
        if "y" in features:
            _features.append(1)
        if "z" in features:
            _features.append(2)
        if "c" in features:
            _features.append(3)
        self.features = np.array(_features, dtype=np.int32)
        assert self.features.shape[0] > 0, f"{self.features}"

    def __call__(self,
                 data):
        feature = data["feature"]
        # `[C, T, J]`
        feature = feature[self.features]
        feature = feature[:, :, self.landmarks]
        data["feature"] = feature
        return data


class PartsBasedNormalization():
    def __init__(self,
                 pose_head=0,
                 pose_num=12,
                 pose_origin=[11, 12],
                 pose_unit1=[11],
                 pose_unit2=[12],
                 lhand_head=12,
                 lhand_num=21,
                 lhand_origin=[0, 2, 5, 9, 13, 17],
                 lhand_unit1=[0],
                 lhand_unit2=[2, 5, 9, 13, 17],
                 rhand_head=12+21,
                 rhand_num=21,
                 rhand_origin=[0, 2, 3, 9, 13, 17],
                 rhand_unit1=[0],
                 rhand_unit2=[2, 5, 9, 13, 17],
                 face_head=12+21+21,
                 face_num=76,
                 face_origin=[0, 2],
                 face_unit1=[7],
                 face_unit2=[42]):

        self.pose_head = pose_head
        self.pose_num = pose_num
        self.pose_origin = pose_origin
        self.pose_unit1 = pose_unit1
        self.pose_unit2 = pose_unit2

        self.lhand_head = lhand_head
        self.lhand_num = lhand_num
        self.lhand_origin = lhand_origin
        self.lhand_unit1 = lhand_unit1
        self.lhand_unit2 = lhand_unit2

        self.rhand_head = rhand_head
        self.rhand_num = rhand_num
        self.rhand_origin = rhand_origin
        self.rhand_unit1 = rhand_unit1
        self.rhand_unit2 = rhand_unit2

        self.face_head = face_head
        self.face_num = face_num
        self.face_origin = face_origin
        self.face_unit1 = face_unit1
        self.face_unit2 = face_unit2

    def _gen_tmask(self, feature):
        tmask = feature == 0.0
        tmask = np.all(tmask, axis=(0, 2))
        tmask = np.logical_not(tmask.reshape([1, -1, 1]))
        return tmask

    def _calc_origin(self, feature, origin_lm):
        # `[C, T, J] -> [C, T, 1]`
        origin = feature[:, :, origin_lm].mean(axis=-1, keepdims=True)
        return origin

    def _calc_unit(self, feature, unit_lm1, unit_lm2, unit_range):
        # The frame-wise unit lengths are unstable.
        # So, we calculate average unit length.
        # Extract.
        # `[C, T, J] -> [C, T, 1]`
        unit1 = feature[:, :, unit_lm1].mean(axis=-1)
        unit2 = feature[:, :, unit_lm2].mean(axis=-1)
        # Mean square between target points.
        unit = np.sqrt((unit1 - unit2) ** 2)
        # Norm.
        # `[C, T, J] -> [1, T, 1]`
        unit = np.linalg.norm(unit, axis=0)
        unit = unit.reshape([1, unit.shape[0], 1])
        unit[unit <= 0] = 1.0
        unit[np.isnan(unit)] = 1.0
        # Finally, clip extreme values.
        unit = np.clip(unit, a_min=unit_range[0], a_max=unit_range[1])
        return unit

    def _normalize(self, feature, origin_lm, unit_lm1, unit_lm2,
                   unit_range=[1.0e-3, 5.0]):
        tmask = self._gen_tmask(feature)
        origin = self._calc_origin(feature, origin_lm)
        unit = self._calc_unit(feature, unit_lm1, unit_lm2, unit_range)

        _feature = feature - origin
        _feature = _feature / unit
        _feature = _feature * tmask
        return _feature

    def __call__(self,
                 data):
        feature = data["feature"]
        if self.pose_num > 0:
            pose = feature[:, :, self.pose_head: self.pose_head+self.pose_num]
            pose = self._normalize(pose, self.pose_origin,
                                   self.pose_unit1, self.pose_unit2)
            feature[:, :, self.pose_head: self.pose_head+self.pose_num] = pose
        if self.lhand_num > 0:
            lhand = feature[:, :,
                            self.lhand_head: self.lhand_head+self.lhand_num]
            lhand = self._normalize(lhand, self.lhand_origin,
                                    self.lhand_unit1, self.lhand_unit2)
            feature[:, :, self.lhand_head: self.lhand_head +
                    self.lhand_num] = lhand
        if self.rhand_num > 0:
            rhand = feature[:, :,
                            self.rhand_head: self.rhand_head+self.rhand_num]
            rhand = self._normalize(rhand, self.rhand_origin,
                                    self.rhand_unit1, self.rhand_unit2)
            feature[:, :, self.rhand_head: self.rhand_head +
                    self.rhand_num] = rhand
        if self.face_num > 0:
            face = feature[:, :, self.face_head: self.face_head+self.face_num]
            face = self._normalize(face, self.face_origin,
                                   self.face_unit1, self.face_unit2)
            feature[:, :, self.face_head: self.face_head+self.face_num] = face
        data["feature"] = feature
        return data


class ToTensor():
    """ Convert data to torch.Tensor.
    """

    def __init__(self):
        pass

    def __call__(self,
                 data):
        new_data = {}
        for key, val in data.items():
            if val is not None:
                if isinstance(val, list):
                    for i, subval in enumerate(val):
                        if subval.dtype in [float, np.float64]:
                            val[i] = torch.from_numpy(
                                subval.astype(np.float32))
                        else:
                            val[i] = torch.from_numpy(subval)
                elif isinstance(val, np.ndarray):
                    if val.dtype in [float, np.float64]:
                        val = torch.from_numpy(val.astype(np.float32))
                    else:
                        val = torch.from_numpy(val)
            new_data[key] = val
        return new_data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def get_fullbody_landmarks():
    use_landmarks = np.concatenate(
        [USE_POSE, USE_LHAND, USE_RHAND, USE_FACE]).astype(np.int32)
    use_landmarks = np.sort(use_landmarks)
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks


def get_parts_connections(use_landmarks, parts_name):
    target_connections = None
    if parts_name == "pose":
        target_connections = POSE_CONNECTIONS
    elif parts_name == "lhand":
        target_connections = LHAND_CONNECTIONS
    elif parts_name == "rhand":
        target_connections = RHAND_CONNECTIONS
    elif parts_name == "face":
        target_connections = FACE_CONNECTIONS
    assert target_connections is not None

    connections = np.array(
        [[pair[0], pair[1]] for pair in target_connections
         if pair[0] in use_landmarks and pair[1] in use_landmarks])
    connections_sorted = np.array([[np.where(pair[0] == use_landmarks)[0][0],
                                    np.where(pair[1] == use_landmarks)[0][0]]
                                   for pair in connections])
    return connections, connections_sorted


def draw_landmarks(draw, landmarks, connections,
                   pt_color=(0, 0, 0), line_color=(0, 0, 0),
                   pt_size=3, line_size=3,
                   do_remove_error=True):
    for i, line in enumerate(connections):
        p0 = landmarks[line[0], :2]
        p1 = landmarks[line[1], :2]
        if do_remove_error:
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            if (p0 == 0).any() or (p1 == 0).any():
                continue
        p0 = (int(p0[0]), int(p0[1]))
        p1 = (int(p1[0]), int(p1[1]))
        cv2.line(draw, p0, p1, line_color, line_size)
        cv2.circle(draw, p0, pt_size, pt_color, -1)
        cv2.circle(draw, p1, pt_size, pt_color, -1)
    return draw


def draw_tracking(track_data,
                  out_path,
                  pose_connections,
                  lhand_connections,
                  rhand_connections,
                  face_connections,
                  width=500,
                  height=500,
                  fps=30):
    track_data[np.isnan(track_data)] = 0
    # If base width is smaller than 10, the track_data should be normalized.
    # So, we project them into image plane.
    if np.abs(track_data[:, :, :, 0].max() - track_data[:, :, :, 0].min()) < 10:
        track_data[:, :, :, 0] *= width
        track_data[:, :, :, 1] *= height
    xmin_orig = track_data[:, :, :, 0].min()
    ymin_orig = track_data[:, :, :, 1].min()
    xmax_orig = track_data[:, :, :, 0].max()
    ymax_orig = track_data[:, :, :, 1].max()
    xmin_proj = int(xmin_orig)
    ymin_proj = int(ymin_orig)
    xmax_proj = int(xmax_orig)
    ymax_proj = int(ymax_orig)

    # Add offset.
    offset_x = 0
    offset_y = 0
    if xmin_proj < 0:
        offset_x = -xmin_proj
        track_data[:, :, :, 0] += offset_x
    if ymin_proj < 0:
        offset_y = -ymin_proj
        track_data[:, :, :, 1] += offset_y

    y_win = int(max(ymax_proj + offset_y, height + offset_y))
    x_win = int(max(xmax_proj + offset_x, width + offset_x))

    print("Window size:", x_win, y_win)
    print("Offsets:", offset_x, offset_y)

    # Detect codecs.
    fourcc = None
    if out_path.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    elif out_path.endswith(".webm"):
        fourcc = cv2.VideoWriter_fourcc(*"vp90")  # type: ignore
    assert fourcc is not None

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (x_win, y_win))
    if writer.isOpened() is False:
        print("VideoWriter failed to open.")
        if writer is not None:
            writer.release()
        return

    frame_count = 0
    instances = track_data.shape[0]
    while True:
        image = np.full([height, width, 3], 255, dtype=np.uint8)
        # Draw
        draw = np.full([int(y_win), int(x_win), 3], 64, dtype=np.uint8)
        draw[offset_y: offset_y+width, offset_x: offset_x+width, :] = image

        for i in range(instances):
            frame = track_data[i, frame_count]

            if pose_connections is not None:
                draw = draw_landmarks(draw, frame, pose_connections, [
                                      0, 0, 255], [0, 0, 255])
            if face_connections is not None:
                draw = draw_landmarks(draw, frame, face_connections, [
                                      0, 255, 0], [0, 255, 0])
            if lhand_connections is not None:
                draw = draw_landmarks(draw, frame, lhand_connections, [
                                      255, 0, 0], [255, 0, 0])
            if rhand_connections is not None:
                draw = draw_landmarks(draw, frame, rhand_connections, [
                                      255, 0, 255], [255, 0, 255])
        writer.write(draw)
        frame_count += 1
        if frame_count >= track_data.shape[1]:
            break
    writer.release()


def main(args: Namespace, parser: ArgumentParser):
    try:
        data_dir = args.data_dir
        out_video_select = args.out_video_select
        out_video_normalized = args.out_video_normalized
        out_plot_select = args.out_plot_select
        out_plot_normalized = args.out_plot_normalized
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    track_dir = os.path.join(data_dir, "tracks", "train")
    label_dir = os.path.join(data_dir, "labels", "train")

    assert os.path.isdir(track_dir)
    assert os.path.isdir(label_dir)

    track_files = os.listdir(track_dir)
    label_files = os.listdir(label_dir)

    # -------------------------------------------------
    # Fetch first data.
    # -------------------------------------------------
    track = np.load(os.path.join(track_dir, track_files[0]))
    with open(os.path.join(label_dir, label_files[0]), "r") as fread:
        label = json.load(fread)

    print("The contents of track data:", track)
    print("The shape of track data:", track.shape)
    print("The content of annotation:", label)

    # -------------------------------------------------
    # Fetch data from WlaslDataset.
    # -------------------------------------------------
    label_dictionary = os.path.join(data_dir, "wlasl_100.json")
    with open(label_dictionary, "r") as fread:
        dictionary = json.load(fread)

    key2token = {key: val["token"] for key, val in dictionary.items()}
    token2key = {val: key for key, val in key2token.items()}

    dataset = WlaslDataset(track_dir, label_dir,
                           label_dictionary, convert_to_channel_first=True)

    sample = next(iter(dataset))
    track = sample["feature"]
    tokens = sample["tokens"]
    print("The content of feature:", track)
    print("The shape of feature:", track.shape)
    print("The content of tokens:", tokens)
    print("The meanings of tokens:", token2key[int(tokens.item())])

    # -------------------------------------------------
    # Fetch a datum from DataLoader.
    # -------------------------------------------------
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    sample = next(iter(dataloader))
    track = sample["feature"]
    tokens = sample["tokens"]
    print("The shape of track:", track.shape)
    print("The content of tokens:", tokens)

    sample = next(iter(dataloader))
    track = sample["feature"]
    tokens = sample["tokens"]
    print("The shape of track:", track.shape)
    print("The content of tokens:", tokens)

    # -------------------------------------------------
    # Fetching should be failed when the padding is not applied.
    # -------------------------------------------------
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    try:
        sample = next(iter(dataloader))
    except Exception as inst:
        print(type(inst))
        print(inst)

    # -------------------------------------------------
    # Fetching should be passed when the padding is applied.
    # -------------------------------------------------
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, collate_fn=merge_padded_batch)

    sample = next(iter(dataloader))
    tracks = sample["feature"]
    tracks_pad_mask = sample["feature_pad_mask"]
    tokens = sample["tokens"]
    print("The content of tracks:", tracks)
    print("The shape of tracks:", tracks.shape)
    print("The content of mask:", tracks_pad_mask)
    print("The content of tokens:", tokens)

    # -------------------------------------------------
    # Apply landmark selection.
    # -------------------------------------------------
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero()
    ])

    # Search maximum length.
    max_length = 0
    max_index = -1
    for i, fin in enumerate(track_files):
        track = np.load(os.path.join(track_dir, fin))
        if max_length < track.shape[1]:
            max_length = track.shape[1]
            max_index = i

    track = np.load(os.path.join(track_dir, track_files[max_index]))
    # `[T, J, C] -> [C, T, J]`
    data = {"feature": track[0].transpose([2, 0, 1])}

    data_trans1 = transforms(data)
    feature1 = data_trans1["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of extracted feature:", feature1.shape)

    # -------------------------------------------------
    # Apply landmark selection and normalization.
    # -------------------------------------------------
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
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
            lhand_origin=[0, 2, 5, 9, 13, 17],  # Center of parm
            lhand_unit1=[0],
            lhand_unit2=[2, 5, 9, 13, 17],
            rhand_head=len(USE_POSE) + len(USE_LHAND),
            rhand_num=len(USE_RHAND),
            rhand_origin=[0, 2, 3, 9, 13, 17],  # Center of parm
            rhand_unit1=[0],
            rhand_unit2=[2, 5, 9, 13, 17],
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND),
            face_num=len(USE_FACE),
            face_origin=[0, 2],
            face_unit1=[7],
            face_unit2=[42]),
    ])

    # Search maximum length.
    track = np.load(os.path.join(track_dir, track_files[max_index]))
    # `[T, J, C] -> [C, T, J]`
    data = {"feature": track[0].transpose([2, 0, 1])}

    data_trans2 = transforms(data)
    feature2 = data_trans2["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of extracted feature:", feature2.shape)

    # -------------------------------------------------
    # Draw result of landmark selection.
    # -------------------------------------------------
    _, pose_connections = get_parts_connections(use_landmarks, "pose")
    _, lhand_connections = get_parts_connections(use_landmarks, "lhand")
    _, rhand_connections = get_parts_connections(use_landmarks, "rhand")
    _, face_connections = get_parts_connections(use_landmarks, "face")

    track_extracted = feature1.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_extracted = track_extracted.transpose([1, 2, 0])
    track_extracted = np.expand_dims(track_extracted, axis=0)
    print(track_extracted.shape)
    draw_tracking(track_extracted, out_video_select,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # -------------------------------------------------
    # Draw result of normalization.
    # -------------------------------------------------
    track_normalized = feature2.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_normalized = track_normalized.transpose([1, 2, 0])
    track_normalized = np.expand_dims(track_normalized, axis=0)
    print(track_normalized.shape)
    draw_tracking(track_normalized, out_video_normalized,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=200,
                  height=200)

    # -------------------------------------------------
    # Plot result of landmark selection.
    # -------------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans1 = feature1[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_trans1.shape)

    # Draw x values.
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(1, 2)  # 2 columns.

    # Draw full joints.
    im_orig = axes[0].imshow(np.flipud(xs_orig), cmap=plt.cm.Blues,
                             extent=[0, xs_orig.shape[1], 0, xs_orig.shape[0]])
    im_orig.set_clim(-0.25, 1.25)
    axes[0].set_xlabel("Frames", fontsize=20)
    axes[0].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_orig.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    # Draw extracted joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans1), cmap=plt.cm.Blues,
                              extent=[0, xs_trans1.shape[1], 0, xs_trans1.shape[0]])
    im_trans.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans1.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig(out_plot_select)
    plt.close()

    # -------------------------------------------------
    # Plot result of normalization.
    # -------------------------------------------------
    # Extract x values.
    xs_trans1 = feature1[0, :, :].T
    xs_trans2 = feature2[0, :, :].T
    print("The shape of extracted track data:", xs_trans1.shape)
    print("The shape of normalized track data:", xs_trans2.shape)

    # Draw x values.
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(1, 2)  # 2 columns.

    # Draw full joints.
    im_trans1 = axes[0].imshow(np.flipud(xs_trans1), cmap=plt.cm.Blues,
                               extent=[0, xs_trans1.shape[1], 0, xs_trans1.shape[0]])
    im_trans1.set_clim(-0.25, 1.25)
    axes[0].set_xlabel("Frames", fontsize=20)
    axes[0].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans1.shape[1])
    axes[1].set_ylim(0, xs_trans1.shape[0])

    # Draw extracted joints.
    im_trans2 = axes[1].imshow(np.flipud(xs_trans2), cmap=plt.cm.Blues,
                               extent=[0, xs_trans2.shape[1], 0, xs_trans2.shape[0]])

    min_val = xs_trans2.min()
    max_val = xs_trans2.max()
    range = max_val - min_val
    im_trans2.set_clim(min_val - (range * 0.25), max_val + (range * 0.25))
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans2.shape[1])
    axes[1].set_ylim(0, xs_trans2.shape[0])

    fig.colorbar(im_trans1)
    fig.colorbar(im_trans2)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig(out_plot_normalized)
    plt.close()

    # -------------------------------------------------
    # Apply transformation into DataLoader.
    # -------------------------------------------------
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
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
            lhand_origin=[0, 2, 5, 9, 13, 17],  # Center of parm
            lhand_unit1=[0],
            lhand_unit2=[2, 5, 9, 13, 17],
            rhand_head=len(USE_POSE) + len(USE_LHAND),
            rhand_num=len(USE_RHAND),
            rhand_origin=[0, 2, 3, 9, 13, 17],  # Center of parm
            rhand_unit1=[0],
            rhand_unit2=[2, 5, 9, 13, 17],
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND),
            face_num=len(USE_FACE),
            face_origin=[0, 2],
            face_unit1=[7],
            face_unit2=[42]),
        ToTensor()
    ])

    dataset = WlaslDataset(track_dir, label_dir, label_dictionary,
                           transforms=transforms, convert_to_channel_first=True)

    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, collate_fn=merge_padded_batch)

    sample = next(iter(dataloader))
    tracks = sample["feature"]
    tracks_pad_mask = sample["feature_pad_mask"]
    tokens = sample["tokens"]
    print("The content of tracks:", tracks)
    print("The shape of tracks:", tracks.shape)
    print("The content of mask:", tracks_pad_mask)
    print("The content of tokens:", tokens)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--out_video_select", type=str,
                        default="out_feature1.mp4")
    parser.add_argument("--out_video_normalized", type=str,
                        default="out_feature2.mp4")
    parser.add_argument("--out_plot_select", type=str,
                        default="out_feature1.png")
    parser.add_argument("--out_plot_normalized", type=str,
                        default="out_feature2.png")

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
