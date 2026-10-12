#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_1_2_video_recorder: Gradio based video recorder application.
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
import shutil

from argparse import (
    ArgumentParser,
    Namespace
)
from datetime import datetime


# Third party's modules
import gradio as gr

# Local modules

VIDEO_DIR = ""


def copy_video(video, video_name, add_timestamp):
    global VIDEO_DIR
    if not video_name.strip():
        raise gr.Error("動画ファイル名を入力してください．")
    if video_name == "":
        raise gr.Error("動画ファイル名を入力してください．")

    if add_timestamp:
        date = datetime.now()
        if ".mp4" in video_name:
            temp = video_name.split(".mp4")[0]
            outfile = f"{temp}_{date.strftime('%Y_%m_%d_%H_%M_%S')}.mp4"
        else:
            outfile = f"{video_name}_{date.strftime('%Y_%m_%d_%H_%M_%S')}.mp4"
    else:
        outfile = video_name if ".mp4" in video_name else video_name + ".mp4"
    out_path = os.path.join(VIDEO_DIR, outfile)
    shutil.copyfile(video, out_path)
    return out_path


def main(args: Namespace, parser: ArgumentParser):
    try:
        video_dir = args.video_dir
        port = args.port
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    global VIDEO_DIR
    VIDEO_DIR = video_dir
    os.makedirs(VIDEO_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        with gr.Row():
            # 左カラムのUI
            with gr.Column():
                input_video = gr.Video(sources=["webcam"], format="mp4")
                video_name = gr.Textbox(label="録画ファイル名")
                add_timestamp = gr.Checkbox(label="ファイル名に日付を追加")
            # 右カラムのUI
            with gr.Column():
                output_video = gr.Video(format="mp4")
        # イベントの登録
        input_video.stop_recording(fn=copy_video,
                                   inputs=[input_video,
                                           video_name,
                                           add_timestamp],
                                   outputs=[output_video])

    demo.launch(debug=True, allowed_paths=[VIDEO_DIR], server_port=port)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("--video-dir", default="./videos")
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
