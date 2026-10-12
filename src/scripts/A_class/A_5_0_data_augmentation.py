#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A_5_0_data_augmentation: Data augmentation.
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
import copy
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

from matplotlib import pyplot as plt
from torchvision.transforms import Compose

# Local modules
from modules.defines import (
    USE_POSE,
    USE_LHAND,
    USE_RHAND,
    USE_FACE,
    get_fullbody_landmarks
)
from modules.transforms import (
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    PartsBasedNormalization)

matplotlib.use("Agg")

SWAPPAIR_POSE = np.array([
    [1, 4], [2, 5], [3, 6], [7, 8], [9, 10],
    [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]])

SWAPPAIR_HANDS = np.array([[pl, pr] for pl, pr in zip(USE_LHAND, USE_RHAND)])

# Pair of landmarks to swap.
SWAPPAIR_LIP = np.array([
    # Upper line in outline.
    [37, 267], [39, 269], [40, 270], [185, 409],
    # Both side in outline.
    [61, 291],
    # Lower line in outline.
    [84, 314], [181, 405], [91, 321], [146, 375],
    # Upper line in inline.
    [82, 312], [81, 311], [80, 310], [191, 415],
    # Both side in inline.
    [78, 308],
    # Lower line in inline.
    [87, 317], [178, 402], [88, 318], [95, 324]])
SWAPPAIR_NOSE = np.array([
    [45, 275], [220, 440], [115, 344], [48, 278], [64, 294],
    [98, 327], [97, 326]])
SWAPPAIR_EYE = np.array([
    # Upper line.
    [173, 398], [157, 384], [158, 385], [159, 386], [160, 387],
    [161, 388], [246, 466],
    # Both corner.
    [133, 362], [33, 263],
    # Lower line.
    [155, 382], [154, 381], [153, 380], [145, 374], [144, 373],
    [163, 390], [7, 249]])
SWAPPAIR_FACE = np.concatenate(
    [SWAPPAIR_LIP, SWAPPAIR_NOSE, SWAPPAIR_EYE], axis=0) + 33 + 21 + 21


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


def get_fullbody_swappairs(do_sort=True):
    _, use_landmarks = get_fullbody_landmarks(do_sort=do_sort)
    swappair = np.concatenate(
        [SWAPPAIR_POSE, SWAPPAIR_HANDS, SWAPPAIR_FACE], axis=0)
    swappair_filtered = np.array([
        [pt[0], pt[1]] for pt in swappair
        if pt[0] in use_landmarks and pt[1] in use_landmarks])
    swappair_filtered = np.concatenate(
        [np.where(pt == use_landmarks)[0]
         for pt in swappair_filtered.flatten()])
    swappair_filtered = swappair_filtered.reshape([-1, 2])
    return swappair_filtered, swappair


class RandomHorizontalFlip():
    def __init__(self,
                 apply_ratio,
                 num_joints,
                 swap_pairs,
                 flip_center=0.5,
                 feature_dim=2,
                 include_conf=True):
        assert swap_pairs.shape[1] == 2, f"{swap_pairs}"
        self.apply_ratio = apply_ratio
        self.flip_center = flip_center
        self.feature_dim = feature_dim + 1 if include_conf else feature_dim

        indices = np.arange(0, num_joints, 1)
        for pair in swap_pairs:
            indices[pair[0]] = pair[1]
            indices[pair[1]] = pair[0]
        self.swap_indices = indices

    def __call__(self,
                 data):
        """Execute flipping.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        # Flip positions.
        temp = copy.deepcopy(feature)
        shape = temp.shape
        # `[C, T, J] -> [C, T*J]`
        temp = temp.reshape([self.feature_dim, -1])

        # Filter out when all elements == 0.
        # (it should be confidence == 0).
        row_indices = np.where((temp == 0).all(axis=0) == np.True_)[0]
        temp[0, :] = -temp[0, :] + 2 * self.flip_center
        temp[:, row_indices] = np.array([0] * self.feature_dim)[:, None]
        temp = temp.reshape(shape)
        temp = temp[:, :, self.swap_indices]
        assert temp.shape == feature.shape, f"{temp.shape}, {feature.shape}"

        data["feature"] = temp
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def get_affine_matrix_2d(center,
                         trans,
                         scale,
                         rot,
                         skew,
                         to_radians=True,
                         order=["center", "scale", "rot", "skew", "trans"],
                         dtype=np.float32):
    center = np.array(center)
    trans = np.array(trans)
    scale = np.array(scale)
    center_m = np.array([[1, 0, -center[0]],
                         [0, 1, -center[1]],
                         [0, 0, 1]])
    scale_m = np.array([[scale[0], 0, 0],
                        [0, scale[1], 0],
                        [0, 0, 1]])
    rot = np.radians(rot) if to_radians else rot
    _cos = np.cos(rot)
    _sin = np.sin(rot)
    rot_m = np.array([[_cos, -_sin, 0],
                      [_sin, _cos, 0],
                      [0, 0, 1]])
    _tan = np.tan(np.radians(skew)) if to_radians else np.tan(skew)
    skew_m = np.array([[1, _tan[0], 0],
                       [_tan[1], 1, 0],
                       [0, 0, 1]])
    move = center + trans
    trans_m = np.array([[1, 0, move[0]],
                        [0, 1, move[1]],
                        [0, 0, 1]])

    mat = np.identity(3, dtype=dtype)
    for name in order:
        if name == "center":
            mat = np.matmul(center_m, mat)
        if name == "scale":
            mat = np.matmul(scale_m, mat)
        if name == "rot":
            mat = np.matmul(rot_m, mat)
        if name == "skew":
            mat = np.matmul(skew_m, mat)
        if name == "trans":
            mat = np.matmul(trans_m, mat)
    return mat.astype(dtype)


def apply_affine(inputs, mat, channel_first=True):
    if channel_first:
        # `[C, T, J] -> [T, J, C]`
        inputs = inputs.transpose([1, 2, 0])
    xy = inputs[:, :, :2]
    xy = np.concatenate([xy, np.ones([xy.shape[0], xy.shape[1], 1])], axis=-1)
    xy = np.einsum("...j,ij", xy, mat)
    inputs[:, :, :2] = xy[:, :, :-1]
    if channel_first:
        # `[T, J, C] -> [C, T, J]`
        inputs = inputs.transpose([2, 0, 1])
    return inputs


class RandomAffineTransform2D():
    def __init__(self,
                 apply_ratio,
                 center_joints,
                 target_joints,
                 trans_range,
                 scale_range,
                 rot_range,
                 skew_range,
                 channel_first=True,
                 apply_post_mask=True,
                 random_seed=None,
                 order=["center", "scale", "rot", "skew", "trans"],
                 dtype=np.float32):
        self.apply_ratio = apply_ratio
        self.center_joints = center_joints
        self.target_joints = target_joints
        self.trans_range = trans_range
        self.scale_range = scale_range
        self.rot_range = np.radians(rot_range)
        self.skew_range = np.radians(skew_range)
        self.order = order
        self.dtype = dtype
        self.channel_first = channel_first
        self.apply_post_mask = apply_post_mask
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()

    def __call__(self,
                 data):
        if self.rng.uniform() > self.apply_ratio:
            return data
        # `[C, T, J]`
        feature = data["feature"]

        # Calculate center position.
        temp = feature[:, :, self.center_joints]
        mask = np.sum(temp, axis=(0, 2)) != 0
        if np.all(mask == np.False_):
            return data

        # Use x and y only.
        # `[C, T, J] -> [C, J] -> [C]`
        center = temp[:, mask].mean(axis=1).mean(axis=1)[:2]

        trans = self.rng.uniform(self.trans_range[0], self.trans_range[1], 2)
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1], 2)
        rot = self.rng.uniform(self.rot_range[0], self.rot_range[1])
        skew = self.rng.uniform(self.skew_range[0], self.skew_range[1], 2)

        # Calculate matrix.
        mat = get_affine_matrix_2d(center, trans, scale, rot, skew,
                                   to_radians=False, order=self.order,
                                   dtype=self.dtype)

        # Apply transform.
        if self.apply_post_mask:
            temp = feature.reshape([feature.shape[0], -1])
            mask = (temp == 0).all(axis=0).reshape(
                [1, feature.shape[1], feature.shape[2]])
            mask = np.bitwise_not(mask).astype(feature.dtype)
        else:
            mask = None
        target = feature[:, :, self.target_joints]
        target = apply_affine(target, mat, self.channel_first)
        feature[:, :, self.target_joints] = target
        if mask is not None:
            feature *= mask
        data["feature"] = feature
        return data


class RandomClip():
    def __init__(self,
                 apply_ratio,
                 clip_range,
                 offset=-1,
                 min_apply_size=10):
        assert apply_ratio >= 0.0 and apply_ratio <= 1.0, f"{apply_ratio}"
        assert clip_range[0] <= clip_range[1], f"{clip_range}"
        assert clip_range[0] >= 0.0 and clip_range[0] < 1.0, f"{clip_range}"
        assert clip_range[1] >= 0.0 and clip_range[1] < 1.0, f"{clip_range}"
        self.apply_ratio = apply_ratio
        self.clip_range = clip_range

        self.offset = offset
        self.min_apply_size = min_apply_size

    def __call__(self,
                 data):
        """Execute clipping.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        base_time_len = feature.shape[1]
        aug_time_len = base_time_len
        if base_time_len > self.min_apply_size:
            aug_t_scale = np.random.random(
            ) * (self.clip_range[1] - self.clip_range[0]) + self.clip_range[0]
            aug_time_len = int(base_time_len * (1 - aug_t_scale))
            if aug_time_len < base_time_len:
                if self.offset < 0:
                    offset = np.random.randint(0, base_time_len - aug_time_len)
                else:
                    offset = self.offset
                # `[C, T, J]`
                tail = max(min(aug_time_len + offset,
                           base_time_len-1), offset+1)
                feature = feature[:, offset: tail, :]
        message = f"{feature.shape}, {base_time_len}, {aug_time_len}, " \
            + "{offset}, {tail}"
        assert feature.shape[1] > 0, message
        data["feature"] = feature
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def matrix_interp(x, xs, ys):
    orig_shape = ys.shape

    # ========================================================================
    # Interpolation.
    # ========================================================================
    xs = xs.astype(ys.dtype)
    x = x.astype(ys.dtype)
    # Pad control points for extrapolation.
    xs = np.concatenate([[np.finfo(xs.dtype).min], xs, [
                        np.finfo(xs.dtype).max]], axis=0)
    ys = np.concatenate([ys[:1], ys, ys[-1:]], axis=0)

    # Compute slopes, pad at the edges to flatten.
    sloops = (ys[1:] - ys[:-1]) / np.expand_dims((xs[1:] - xs[:-1]), axis=-1)
    sloops = np.pad(sloops[:-1], [(1, 1), (0, 0)])

    # Solve for intercepts.
    intercepts = ys - sloops * np.expand_dims(xs, axis=-1)

    # Search for the line parameters at each input data point.
    # Create a grid of the inputs and piece breakpoints for thresholding.
    # Rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors.
    idx = np.argmax(np.expand_dims(xs, axis=-2) >
                    np.expand_dims(x, axis=-1), axis=-1)
    sloop = sloops[idx]
    intercept = intercepts[idx]

    # Apply the linear mapping at each input data point.
    y = sloop * np.expand_dims(x, axis=-1) + intercept
    y = y.astype(ys.dtype)
    y = y.reshape([-1, orig_shape[1]])
    return y


class RandomTimeWarping():
    def __init__(self,
                 apply_ratio,
                 scale_range,
                 min_apply_size,
                 apply_post_mask=False):
        self.apply_ratio = apply_ratio
        self.scale_range = scale_range
        self.min_apply_size = min_apply_size
        self.apply_post_mask = apply_post_mask

    def __call__(self,
                 data):
        """Apply time warping to data.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        if self.apply_post_mask:
            temp = feature.reshape([feature.shape[0], -1])
            mask = (temp == 0).all(axis=0).reshape(
                [1, feature.shape[1], feature.shape[2]])
            mask = np.bitwise_not(mask).astype(feature.dtype)
        else:
            mask = None
        orig_shape = feature.shape
        t_len = orig_shape[1]
        if t_len > self.min_apply_size:
            # `[C, T, J] -> [T, C*J]`
            feature = feature.transpose([1, 0, 2]).reshape([t_len, -1])

            assert len(feature.shape) == 2, f"{feature.shape}"
            range = self.scale_range[1] - self.scale_range[0]
            aug_scale = np.random.random() * (range) + self.scale_range[0]
            aug_t_len = int((t_len-1) * aug_scale)

            x = np.linspace(0, t_len-1, num=aug_t_len)
            xs = np.arange(t_len)
            new_feature = matrix_interp(x, xs, feature)
            # `[T, C*J] -> [T, C, J] -> [C, T, J]`
            new_feature = new_feature.reshape(
                [-1, orig_shape[0], orig_shape[2]])
            new_feature = new_feature.transpose([1, 0, 2])
            if mask is not None:
                # `[1, T, J] -> [T, 1*J]`
                mask = mask.transpose([1, 0, 2]).reshape([t_len, -1])
                assert len(mask.shape) == 2, f"{mask.shape}"
                new_mask = matrix_interp(x, xs, mask)
                # `[T, 1*J] -> [T, 1, J] -> [1, T, J]`
                new_mask = new_mask.reshape([-1, 1, orig_shape[2]])
                new_mask = new_mask.transpose([1, 0, 2])
                # `[1, T, J] -> [C, T, J]`
                new_mask = np.repeat(new_mask, orig_shape[0], axis=0)
                # Masked out interpolated position.
                new_mask[new_mask < 0.95] = 0.0
                new_feature *= new_mask
        else:
            new_feature = feature

        data["feature"] = new_feature
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def get_parts_connections(use_landmarks, partsname):
    target_connections = None
    if partsname == "pose":
        target_connections = POSE_CONNECTIONS
    elif partsname == "lhand":
        target_connections = LHAND_CONNECTIONS
    elif partsname == "rhand":
        target_connections = RHAND_CONNECTIONS
    elif partsname == "face":
        target_connections = FACE_CONNECTIONS
    assert target_connections is not None
    connections = np.array(
        [[pair[0], pair[1]] for pair in target_connections
         if pair[0] in use_landmarks and pair[1] in use_landmarks])
    connections_sorted = np.array([[np.where(pair[0] == use_landmarks)[0][0],
                                    np.where(pair[1] == use_landmarks)[0][0]]
                                   for pair in connections])
    return connections, connections_sorted


def draw_tracking(track_data,
                  out_path,
                  pose_connections,
                  lhand_connections,
                  rhand_connections,
                  face_connections,
                  width=500,
                  height=500,
                  fps=30,
                  do_remove_error=False):
    track_data[np.isnan(track_data)] = 0
    # If base width is smaller than 10, the trackdata should be normalized.
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
                                      0, 0, 255], [0, 0, 255],
                                      do_remove_error=do_remove_error)
            if face_connections is not None:
                draw = draw_landmarks(draw, frame, face_connections, [
                                      0, 255, 0], [0, 255, 0],
                                      do_remove_error=do_remove_error)
            if lhand_connections is not None:
                draw = draw_landmarks(draw, frame, lhand_connections, [
                                      255, 0, 0], [255, 0, 0],
                                      do_remove_error=do_remove_error)
            if rhand_connections is not None:
                draw = draw_landmarks(draw, frame, rhand_connections, [
                                      255, 0, 255], [255, 0, 255],
                                      do_remove_error=do_remove_error)
        writer.write(draw)
        frame_count += 1
        if frame_count >= track_data.shape[1]:
            break
    writer.release()


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


def main(args: Namespace, parser: ArgumentParser):
    try:
        data_dir = args.data_dir
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    track_dir = os.path.join(data_dir, "tracks", "train")
    label_dir = os.path.join(data_dir, "labels", "train")

    track_files = os.listdir(track_dir)
    label_files = os.listdir(label_dir)

    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    swap_pairs, _ = get_fullbody_swappairs()

    # =======================================================================
    # Without normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomHorizontalFlip(
            apply_ratio=1.0,
            num_joints=len(use_landmarks),
            swap_pairs=swap_pairs,
            flip_center=0.5,
            feature_dim=len(use_features),
            include_conf=False)
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
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomAffineTransform2D(
            apply_ratio=1.0,
            center_joints=[0],  # Nose
            target_joints=np.arange(0, len(use_landmarks)),
            trans_range=[0.1, 0.1],
            scale_range=[1.2, 1.2],
            rot_range=[30, 30],
            skew_range=[5, 5])
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

    data_trans2 = transforms(data)
    feature2 = data_trans2["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomClip(
            apply_ratio=1.0,
            clip_range=[0.2, 0.2],
            min_apply_size=2)
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

    data_trans3 = transforms(data)
    feature3 = data_trans3["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomTimeWarping(
            apply_ratio=1.0,
            scale_range=[1.5, 1.5],
            min_apply_size=2)
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

    data_trans4 = transforms(data)
    feature4 = data_trans4["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomTimeWarping(
            apply_ratio=1.0,
            scale_range=[1.5, 1.5],
            min_apply_size=2,
            apply_post_mask=True)
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

    data_trans5 = transforms(data)
    feature5 = data_trans5["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # =======================================================================
    # With normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    swap_pairs, _ = get_fullbody_swappairs()

    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomHorizontalFlip(
            apply_ratio=1.0,
            num_joints=len(use_landmarks),
            swap_pairs=swap_pairs,
            flip_center=0.5,
            feature_dim=len(use_features),
            include_conf=False),
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
            face_unit2=[42])
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

    data_trans6 = transforms(data)
    feature6 = data_trans6["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomAffineTransform2D(
            apply_ratio=1.0,
            center_joints=[0],  # Nose
            target_joints=np.arange(0, len(use_landmarks)),
            trans_range=[0.1, 0.1],
            scale_range=[1.2, 1.2],
            rot_range=[30, 30],
            skew_range=[5, 5]),
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
            face_unit2=[42])
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

    data_trans7 = transforms(data)
    feature7 = data_trans7["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomClip(
            apply_ratio=1.0,
            clip_range=[0.2, 0.2],
            min_apply_size=2),
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
            face_unit2=[42])
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

    data_trans8 = transforms(data)
    feature8 = data_trans8["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature1.shape)

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomTimeWarping(
            apply_ratio=1.0,
            scale_range=[1.5, 1.5],
            min_apply_size=2),
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
            face_unit2=[42])
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

    data_trans9 = transforms(data)
    feature9 = data_trans9["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature9.shape)

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, features=use_features),
        ReplaceNan(),
        FilterZero(),
        RandomTimeWarping(
            apply_ratio=1.0,
            scale_range=[1.5, 1.5],
            min_apply_size=2,
            apply_post_mask=True),
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
            face_unit2=[42])
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

    data_trans10 = transforms(data)
    feature10 = data_trans10["feature"]
    print("The shape of original feature:", track.shape)
    print("The shape of transformed feature:", feature10.shape)

    # =======================================================================
    # Draw tracking.
    # =======================================================================
    _, pose_connections = get_parts_connections(use_landmarks, "pose")
    _, lhand_connections = get_parts_connections(use_landmarks, "lhand")
    _, rhand_connections = get_parts_connections(use_landmarks, "rhand")
    _, face_connections = get_parts_connections(use_landmarks, "face")

    # =======================================================================
    # Without normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
    output_video = "./out_feature1.mp4"
    track_transformed = feature1.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    output_video = "./out_feature2.mp4"
    track_transformed = feature2.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    output_video = "./out_feature3.mp4"
    track_transformed = feature3.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    output_video = "./out_feature4.mp4"
    track_transformed = feature4.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    output_video = "./out_feature5.mp4"
    track_transformed = feature5.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # =======================================================================
    # With normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
    output_video = "./out_feature6.mp4"
    track_transformed = feature6.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    output_video = "./out_feature7.mp4"
    track_transformed = feature7.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    output_video = "./out_feature8.mp4"
    track_transformed = feature8.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    output_video = "./out_feature9.mp4"
    track_transformed = feature9.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    output_video = "./out_feature10.mp4"
    track_transformed = feature10.copy()
    # `[C, T, J] -> [P, T, J, C]`, P == 1
    track_transformed = track_transformed.transpose([1, 2, 0])
    track_transformed = np.expand_dims(track_transformed, axis=0)
    print(track_transformed.shape)
    draw_tracking(track_transformed, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections,
                  width=500,
                  height=500)

    # =======================================================================
    # Plot tracking as feature map.
    # =======================================================================
    # =======================================================================
    # Without normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
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

    plt.savefig("out_feature1.png")
    plt.close()

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans2 = feature2[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_trans2.shape)

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
    im_trans = axes[1].imshow(np.flipud(xs_trans2), cmap=plt.cm.Blues,
                              extent=[0, xs_trans2.shape[1], 0, xs_trans2.shape[0]])
    im_trans.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans2.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature2.png")
    plt.close()

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans3 = feature3[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_trans3.shape)

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
    im_trans = axes[1].imshow(np.flipud(xs_trans3), cmap=plt.cm.Blues,
                              extent=[0, xs_trans3.shape[1], 0, xs_trans3.shape[0]])
    im_trans.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans3.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature3.png")
    plt.close()

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans4 = feature4[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_trans4.shape)

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
    im_trans = axes[1].imshow(np.flipud(xs_trans4), cmap=plt.cm.Blues,
                              extent=[0, xs_trans4.shape[1], 0, xs_trans4.shape[0]])
    im_trans.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans4.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature4.png")
    plt.close()

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans5 = feature5[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_trans5.shape)

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
    im_trans = axes[1].imshow(np.flipud(xs_trans5), cmap=plt.cm.Blues,
                              extent=[0, xs_trans5.shape[1], 0, xs_trans5.shape[0]])
    im_trans.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans5.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature5.png")
    plt.close()

    # =======================================================================
    # With normalization.
    # =======================================================================
    # --------------------------------------------
    # Horizontal flip.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans6 = feature6[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of normalized track data:", xs_trans6.shape)

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

    # Draw normalized joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans6), cmap=plt.cm.Blues,
                              extent=[0, xs_trans6.shape[1], 0, xs_trans6.shape[0]])
    im_trans.set_clim(feature6.min(), feature6.max())
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans6.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature6.png")
    plt.close()

    # --------------------------------------------
    # Affine transform.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans7 = feature7[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of normalized track data:", xs_trans7.shape)

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

    # Draw normalized joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans7), cmap=plt.cm.Blues,
                              extent=[0, xs_trans7.shape[1], 0, xs_trans7.shape[0]])
    im_trans.set_clim(feature7.min(), feature7.max())
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans7.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature7.png")
    plt.close()

    # --------------------------------------------
    # Temporal clipping.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans8 = feature8[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of normalized track data:", xs_trans8.shape)

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

    # Draw normalized joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans8), cmap=plt.cm.Blues,
                              extent=[0, xs_trans8.shape[1], 0, xs_trans8.shape[0]])
    im_trans.set_clim(feature8.min(), feature8.max())
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans8.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature8.png")
    plt.close()

    # --------------------------------------------
    # Temporal warping.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans9 = feature9[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of normalized track data:", xs_trans9.shape)

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

    # Draw normalized joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans9), cmap=plt.cm.Blues,
                              extent=[0, xs_trans9.shape[1], 0, xs_trans9.shape[0]])
    im_trans.set_clim(feature9.min(), feature9.max())
    print(feature9[:, :, 22:22+21].min(), feature9[:, :, 22:22+21].max())
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans9.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature9.png")
    plt.close()

    # --------------------------------------------
    # Temporal warping with masking.
    # --------------------------------------------
    # Extract x values.
    xs_orig = track[0, :, :, 0].T
    xs_trans10 = feature10[0, :, :].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of normalized track data:", xs_trans10.shape)

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

    # Draw normalized joints.
    im_trans = axes[1].imshow(np.flipud(xs_trans10), cmap=plt.cm.Blues,
                              extent=[0, xs_trans10.shape[1], 0, xs_trans10.shape[0]])
    im_trans.set_clim(feature10.min(), feature10.max())
    print(feature10[:, :, 22:22+21].min(), feature10[:, :, 22:22+21].max())
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)
    axes[1].set_xlim(0, xs_trans10.shape[1])
    axes[1].set_ylim(0, xs_orig.shape[0])

    fig.colorbar(im_orig)
    fig.colorbar(im_trans)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig("out_feature10.png")
    plt.close()

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("data_dir", type=str)

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
