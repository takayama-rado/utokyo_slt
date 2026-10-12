#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_Appendix_E2_demo_psj_simple_isolated_slr: Demo application.
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

# Third party's modules
import cv2

import gradio as gr

import mediapipe as mp

import numpy as np

import torch

from torch import nn

from torchvision.transforms import Compose

# Local modules
from modules.defines import (
    USE_POSE,
    USE_LHAND,
    USE_RHAND,
    USE_FACE,
    get_fullbody_landmarks)
from modules.model import (
    GPoolRecognitionHead)
from modules.tracking import (
    track_holistic)
from modules.transforms import (
    SelectLandmarksAndFeature,
    ReplaceNan,
    FilterZero,
    PartsBasedNormalization,
    ToTensor)
from modules.utils import (
    load_checkpoint)

DEVICE = None
HOLISTIC_MODEL_DIR = "./modules/holistic_models"
TRACKER = None
RECOGNIZER = None
TOKENS_TO_WORDS = None
PREPROCESSOR = None


class SimpleIsoSLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear1 = nn.Linear(in_channels, 64)
        self.norm1 = nn.BatchNorm1d(64)

        self.linear2 = nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(64)

        self.activation = nn.ReLU()
        self.head = GPoolRecognitionHead(64, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        """Forward computation.

        # Args:
          - inputs: `[N, in_channels, T, J]`
          - pad_mask: `[N, T]`
        # Returns:
          - logits: `[N, out_channels]`
        """
        # `[N, C, T, J] -> [N, T, J, C] -> [N, T, J*C]`
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])

        # 1st block.
        # `[N, T, C]`
        feature = self.linear1(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.norm1(feature)
        feature = self.activation(feature)

        # 2nd block.
        residual = feature
        # `[N,  C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1])
        feature = self.linear2(feature)
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1])
        feature = self.norm2(feature)
        feature = self.activation(feature)
        # Residual connection.
        feature = feature + residual
        feature = self.activation(feature)

        logits = self.head(feature, feature_pad_mask)
        return logits


def track_and_recognize(video, recognized_words_list, progress=gr.Progress()):
    global TRACKER, RECOGNIZER, PREPROCESSOR, DEVICE, TOKENS_TO_WORDS
    assert TRACKER is not None
    assert RECOGNIZER is not None
    assert PREPROCESSOR is not None
    assert TOKENS_TO_WORDS is not None

    progress(0, desc="動作追跡中...")
    # Start tracking.
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
    landmarks = track_holistic(capture, TRACKER)
    assert landmarks is not None
    print("Done.")
    capture.release()
    progress(0.5, desc="認識中...")

    # Recognize.
    RECOGNIZER.eval()
    with torch.no_grad():
        feature = landmarks[0]  # Single person.
        # `[T, J, C] -> [C, T, J]`
        feature = feature.transpose([2, 0, 1])
        data = {"feature": feature}

        # Preprocess.
        data = PREPROCESSOR(data)
        feature = data["feature"]
        # Add batch axis.
        if len(feature.shape) == 3:
            feature = feature.unsqueeze(0)
        feature = feature.to(DEVICE)

        # Inference.
        pred = RECOGNIZER(feature=feature)
        pred_ids = pred.argmax(dim=1).unsqueeze(-1)

        # Tokens -> words.
        pred_ids = pred_ids.detach().cpu().numpy().squeeze(0)
        pred_ids = [int(pid) for pid in pred_ids]
        words = [TOKENS_TO_WORDS[pid] for pid in pred_ids]
        print("pred_ids:", pred_ids)
        print("words:", words)
    recognized_words_list.append(words)
    return "認識完了", recognized_words_list


def main(args: Namespace, parser: ArgumentParser):
    try:
        data_dir = args.data_dir
        ckpt_dir = args.ckpt_dir
        port = args.port
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    global DEVICE, HOLISTIC_MODEL_DIR, TRACKER, RECOGNIZER, TOKENS_TO_WORDS, PREPROCESSOR

    base_options = mp.tasks.BaseOptions(
        model_asset_path=os.path.join(HOLISTIC_MODEL_DIR,
                                      "holistic_landmarker.task"))
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO)

    TRACKER = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using {DEVICE} for computation.")

    with open(os.path.join(ckpt_dir, "hparams.json"), "r") as fread:
        hparams = json.load(fread)
    print(hparams)

    # Load model.
    in_channels = hparams["in_channels"]
    out_channels = hparams["out_channels"]
    RECOGNIZER = SimpleIsoSLR(in_channels, out_channels)

    RECOGNIZER, _, _, _ = load_checkpoint(
        os.path.join(ckpt_dir, "best_acc.ckpt"), RECOGNIZER)
    RECOGNIZER.to(DEVICE)
    RECOGNIZER.eval()

    # Load dictionary.
    with open(os.path.join(data_dir, "custom_psj.json"), "r") as fread:
        word_dict = json.load(fread)
    TOKENS_TO_WORDS = {val["token"]: val["represents"][0]
                       for val in word_dict.values()}

    # -----------------------------------------------------
    # Setup preprocessor.
    # -----------------------------------------------------
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]
    PREPROCESSOR = Compose([
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

    # -----------------------------------------------------
    # Create application.
    # -----------------------------------------------------
    text_size = gr.themes.Size(  # type: ignore
        name="text_custom",
        xxs="12px",
        xs="14px",
        sm="16px",
        md="18px",
        lg="22px",
        xl="26px",
        xxl="30px")
    theme = gr.themes.Default(text_size=text_size)  # type: ignore

    with gr.Blocks(theme=theme) as demo:
        recognized_words_list = gr.State([])
        with gr.Row():
            with gr.Column():  # Left column.
                input_video = gr.Video(sources=["webcam"], format="mp4")
                clear_button = gr.ClearButton(value="クリア")
            with gr.Column():  # Right column.
                status = gr.Text(label="ステータス", interactive=False)

                @gr.render(inputs=recognized_words_list)
                def render_recognized_words(_words_list):
                    gr.Markdown("認識・翻訳結果")
                    for _words in _words_list:
                        recog_text = "認識結果: "
                        recog_text += "".join(f"[{_word}], " for _word in _words)
                        with gr.Row():
                            gr.Textbox(recog_text, show_label=False,
                                       container=False,
                                       interactive=False)
        # Define events.
        input_video.stop_recording(fn=track_and_recognize,
                                   inputs=[input_video, recognized_words_list],
                                   outputs=[status, recognized_words_list])
        clear_button.add([input_video, recognized_words_list])

    demo.launch(debug=True, server_port=port)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_dir", type=str)
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
