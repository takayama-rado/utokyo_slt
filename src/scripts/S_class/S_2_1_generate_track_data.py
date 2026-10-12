#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_2_1_generate_track_data: Generate tracking data.
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
import os

from argparse import (
    ArgumentParser,
    Namespace
)

# Third party's modules
import cv2

import mediapipe as mp

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Local modules

matplotlib.use("Agg")
HOLISTIC_MODEL_DIR = "./modules/holistic_models"

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


def _ext_holistic_landmarks(result):
    # `[P, T, J, C]`
    body_lmarks = np.zeros([1, 1, 33, 4])
    lhand_lmarks = np.zeros([1, 1, 21, 4])
    rhand_lmarks = np.zeros([1, 1, 21, 4])
    face_lmarks = np.zeros([1, 1, 478, 4])
    if result.pose_landmarks is not None and len(result.pose_landmarks) > 0:
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, lmark.visibility] for lmark
            in result.pose_landmarks])
        body_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks

    # Visibility of left/right hands and face return dummy value (0.0).
    # So, we assign binary visibilities judged by coordinates values.
    def _binary_conf(lmark):
        return float((lmark.x + lmark.y + lmark.z) > 0)

    if (result.left_hand_landmarks is not None
            and len(result.left_hand_landmarks) > 0):
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.left_hand_landmarks])
        lhand_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    if (result.right_hand_landmarks is not None
            and len(result.right_hand_landmarks) > 0):
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.right_hand_landmarks])
        rhand_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    if (result.face_landmarks is not None
            and len(result.face_landmarks) > 0):
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.face_landmarks])
        face_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    return body_lmarks, lhand_lmarks, rhand_lmarks, face_lmarks


def track_holistic(capture, tracker):
    # Start tracking.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    landmarks = None
    count = 0
    ms_per_frame = 33  # 30 FPS.
    while True:
        status, image = capture.read()
        if status is False:
            break

        # Tracking.
        # The OpenCV's default format is BGR, but MediaPipe requires RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = tracker.detect_for_video(mp_image, int(count*ms_per_frame))
        body, lhand, rhand, face = _ext_holistic_landmarks(result)

        landmark = np.concatenate([body, lhand, rhand, face], axis=2)
        if landmarks is None:
            landmarks = landmark
        else:
            landmarks = np.concatenate([landmarks, landmark], axis=1)
        count += 1
    capture.release()
    return landmarks


def gen_track_data(video_path):
    # Load video.
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened() is False:
        if os.path.isfile(video_path) is False:
            raise FileNotFoundError(f"Can not find {video_path}.")
        else:
            raise RuntimeError(
                f"VideoCapture failed to open {video_path}.",
                "cv2 may not link to ffmpeg or codec may be unknown.",
                "Please check installation of OpenCV.")
    # Initialize tracking model.
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(HOLISTIC_MODEL_DIR,
                                          "holistic_landmarker.task")),
        running_mode=VisionRunningMode.VIDEO)

    tracker = HolisticLandmarker.create_from_options(options)

    landmarks = track_holistic(capture, tracker)
    return landmarks


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
                  video_path,
                  out_path,
                  pose_connections,
                  lhand_connections,
                  rhand_connections,
                  face_connections):
    # Load video.
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened() is False:
        if os.path.isfile(video_path) is False:
            raise FileNotFoundError(f"Can not find {video_path}.")
        else:
            raise RuntimeError(f"VideoCapture failed to open {video_path}.",
                               "cv2 may not link to ffmpeg or codec may be"
                               " unknown.",
                               "Please check installation of OpenCV.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

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
        status, image = capture.read()
        if status is False:
            break

        # Draw
        draw = np.full([int(y_win), int(x_win), 3], 64, dtype=np.uint8)
        draw[offset_y: offset_y+height, offset_x: offset_x+width, :] = image

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
    capture.release()


def main(args: Namespace, parser: ArgumentParser):
    try:
        input_video = args.input_video
        output_video = args.output_video
        output_plot = args.output_plot
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    # -------------------------------------------------------
    # Generate tracking data.
    # -------------------------------------------------------
    track_data = gen_track_data(input_video)
    assert isinstance(track_data, np.ndarray)
    print(f"The type of track data: {type(track_data)}")
    print(f"The shape of track data: {track_data.shape}")
    print(f"The contents of track data: {track_data}")

    _, use_landmarks = get_fullbody_landmarks()
    track_to_draw = track_data[:, :, use_landmarks, :]

    _, pose_connections = get_parts_connections(use_landmarks, "pose")
    _, lhand_connections = get_parts_connections(use_landmarks, "lhand")
    _, rhand_connections = get_parts_connections(use_landmarks, "rhand")
    _, face_connections = get_parts_connections(use_landmarks, "face")

    # -------------------------------------------------------
    # Draw tracking result.
    # -------------------------------------------------------
    draw_tracking(track_to_draw.copy(), input_video, output_video,
                  pose_connections=pose_connections,
                  lhand_connections=lhand_connections,
                  rhand_connections=rhand_connections,
                  face_connections=face_connections)

    # -------------------------------------------------------
    # Plot tracking result.
    # -------------------------------------------------------
    # Extract x values.
    xs_orig = track_data[0, :, :, 0].T
    xs_draw = track_to_draw[0, :, :, 0].T
    print("The shape of original track data:", xs_orig.shape)
    print("The shape of extracted track data:", xs_draw.shape)

    # Draw x values.
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(1, 2)  # 2 columns.

    # Draw full joints.
    im_orig = axes[0].imshow(np.flipud(xs_orig), cmap=plt.cm.Blues,
                             extent=[0, xs_orig.shape[1], 0, xs_orig.shape[0]])
    im_orig.set_clim(-0.25, 1.25)
    axes[0].set_xlabel("Frames", fontsize=20)
    axes[0].set_ylabel("Joint indices", fontsize=20)

    # Draw extracted joints.
    im_draw = axes[1].imshow(np.flipud(xs_draw), cmap=plt.cm.Blues,
                             extent=[0, xs_draw.shape[1], 0, xs_draw.shape[0]])
    im_draw.set_clim(-0.25, 1.25)
    axes[1].set_xlabel("Frames", fontsize=20)
    axes[1].set_ylabel("Joint indices", fontsize=20)

    fig.colorbar(im_orig)
    fig.colorbar(im_draw)

    for ax in axes:
        ax.set_anchor("SW")

    plt.savefig(output_plot)
    plt.close()

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("input_video", type=str)
    parser.add_argument("--output-video", type=str, default="out.mp4")
    parser.add_argument("--output-plot", type=str, default="out.png")

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
