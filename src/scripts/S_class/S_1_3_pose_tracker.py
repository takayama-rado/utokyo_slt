#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_1_3_pose_tracker: Gradio based pose tracker application.
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
import gc
import os

from argparse import (
    ArgumentParser,
    Namespace
)

# Third party's modules
import cv2

import gradio as gr

import numpy as np

import mediapipe as mp

# Local modules
from modules.defines import (
    get_fullbody_landmarks,
    get_parts_connections
)
from modules.draw_functions import (
    draw_tracking
)
from modules.tracking import (
    track_holistic,
    track_pose
)


VIDEO_DIR = ""
TRACK_DIR = ""

POSE_MODEL_DIR = os.path.join(os.getcwd(), "modules/pose_models")
HOLISTIC_MODEL_DIR = os.path.join(os.getcwd(), "modules/holistic_models")

TRACKER = None


def track_video(video,
                track_parts,
                track_mode):
    global VIDEO_DIR, TRACK_DIR, TRACKER
    # モード設定
    model_path = None
    model_postfix = None
    if track_parts == "全身":
        model_path = f"{HOLISTIC_MODEL_DIR}/holistic_landmarker.task"
        model_postfix = "whole"
    else:
        if track_mode == "高速":
            model_path = f"{POSE_MODEL_DIR}/pose_landmarker_lite.task"
            model_postfix = "body_lite"
        elif track_mode == "標準":
            model_path = f"{POSE_MODEL_DIR}/pose_landmarker_full.task"
            model_postfix = "body_full"
        elif track_mode == "高精度":
            model_path = f"{POSE_MODEL_DIR}/pose_landmarker_heavy.task"
            model_postfix = "body_heavy"

    assert model_path is not None
    assert model_postfix is not None

    # 古いTRACKERをクリーンアップ
    if TRACKER:
        TRACKER.close()
        TRACKER = None
        gc.collect()

    base_options = mp.tasks.BaseOptions(
        model_asset_path=model_path)
    # 身体追跡モデルをロード
    if track_parts == "身体":
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False)
        TRACKER = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    # 全身追跡モデルをロード
    else:
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO)

        TRACKER = mp.tasks.vision.HolisticLandmarker.create_from_options(
            options)

    # 追跡開始.
    print("Start tracking...")
    capture = cv2.VideoCapture(video)
    if capture.isOpened() is False:
        if os.path.isfile(video) is False:
            raise FileNotFoundError(f"Can not find {video}.")
        else:
            raise RuntimeError(
                f"VideoCapture failed to open {video}.",
                "cv2 may not link to ffmpeg or codec may be unknown.",
                "Please check installation of OpenCV.")
    landmarks = None
    if track_parts == "身体":
        landmarks = track_pose(capture, TRACKER)
    else:
        landmarks = track_holistic(capture, TRACKER)
    print("Done.")
    assert landmarks is not None

    # 追跡データの保存.
    basename = os.path.basename(video).split(".")[0]
    basename = f"{basename}_{model_postfix}"
    out_path = os.path.join(TRACK_DIR, basename + ".npy")
    np.save(out_path, landmarks)

    # 追跡結果の描画.
    print("Draw tracking result.")
    _, use_landmarks = get_fullbody_landmarks()
    track_to_draw = landmarks[:, :, use_landmarks, :]
    _, pose_connections = get_parts_connections(use_landmarks, "pose")
    _, lhand_connections = get_parts_connections(use_landmarks, "lhand")
    _, rhand_connections = get_parts_connections(use_landmarks, "rhand")
    _, face_connections = get_parts_connections(use_landmarks, "face")

    out_path = os.path.join(VIDEO_DIR, basename + ".mp4")
    draw_tracking(track_to_draw, video, out_path,
                  face_connections=face_connections,
                  lhand_connections=lhand_connections,
                  pose_connections=pose_connections,
                  rhand_connections=rhand_connections)
    print("Done.")
    return out_path


def main(args: Namespace, parser: ArgumentParser):
    try:
        video_dir = args.video_dir
        track_dir = args.track_dir
        port = args.port
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    global VIDEO_DIR, TRACK_DIR, POSE_MODEL_DIR, HOLISTIC_MODEL_DIR

    VIDEO_DIR = video_dir
    TRACK_DIR = track_dir

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(TRACK_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        with gr.Row():
            # 左カラムのUI
            with gr.Column():
                input_video = gr.Video()
                track_parts = gr.Radio(["身体", "全身"], value="全身",
                                       label="追跡部位")
                track_mode = gr.Radio(["高速", "標準", "高精度"],
                                      value="標準",
                                      label="追跡モード (身体モードのみ有効)")
                with gr.Row():
                    clear_button = gr.ClearButton(value="クリア")
                    submit_button = gr.Button("実行")
            # 右カラムのUI
            with gr.Column():
                output_video = gr.Video()
        # イベントの登録
        clear_button.add([input_video, output_video])
        submit_button.click(fn=track_video,
                            inputs=[input_video,
                                    track_parts,
                                    track_mode],
                            outputs=[output_video])

    demo.launch(debug=True, allowed_paths=[VIDEO_DIR, TRACK_DIR],
                server_port=port)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("--video-dir", default="./videos")
    parser.add_argument("--track-dir", default="./tracks")
    parser.add_argument("--port", type=int, default=7860)

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
