#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S_1_4_simple_annotator: Gradio based annotation application.
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

from argparse import (
    ArgumentParser,
    Namespace
)

# Third party's modules
import cv2

import gradio as gr


# Local modules


ANNOT_DIR = ""


def create_annotation(video, words):
    global ANNOT_DIR
    # 動画情報の読み込み
    capture = cv2.VideoCapture(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = 1 / fps * frames
    filename = os.path.basename(video).split(".")[0] + ".json"

    data = {
        "filename": filename,
        "targetVideo": os.path.basename(video),
        "videoWidth": width,
        "videoHeight": height,
        "videoDuration": duration
    }

    step = duration / len(words)
    items = []
    for i, word in enumerate(words):
        if i == 0:
            item = {"content": word,
                    "start": 0,
                    "end": step}
        else:
            item = {"content": word,
                    "start": step * i,
                    "end": min(step * (i+1), duration)}
        items.append(item)
    data["items"] = items
    out_path = os.path.abspath(os.path.join(ANNOT_DIR, filename))
    with open(out_path, "w") as fwrite:
        json.dump(data, fwrite, indent=4)
    gr.Info(f"Annotation is saved into {out_path}.")


def main(args: Namespace, parser: ArgumentParser):
    try:
        annot_dir = args.annot_dir
        port = args.port
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        parser.error("Invalid arguments!!")

    global ANNOT_DIR

    ANNOT_DIR = annot_dir
    os.makedirs(ANNOT_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        words = gr.State([])

        with gr.Row():
            # 左カラムのUI
            with gr.Column():
                input_video = gr.Video()
                with gr.Row():
                    clear_button = gr.ClearButton(value="クリア")
                    submit_button = gr.Button("保存")
            # 右カラムのUI
            with gr.Column():
                new_word = gr.Text(label="手話単語を記入してください",
                                   autofocus=True)

                # 単語を追加すると，動的にUIとイベントを追加
                @gr.render(inputs=words)
                def render_words(_words):
                    gr.Markdown(f"登録手話単語: {len(_words)}")
                    for _word in _words:
                        with gr.Row():
                            gr.Textbox(_word, show_label=False,
                                       container=False)
                            delete_button = gr.Button("Delete",
                                                      scale=0,
                                                      variant="stop")

                            def delete(word=_word):
                                _words.remove(word)
                                return _words
                            delete_button.click(delete, None, [words])

        # クリアイベント
        clear_button.add([input_video, new_word, words])

        # 保存イベント
        def add_word(words, new_word):
            return words + [new_word], ""

        new_word.submit(add_word, [words, new_word], [words, new_word])
        submit_button.click(create_annotation,
                            [input_video, words],
                            None)

    demo.launch(debug=True, allowed_paths=[ANNOT_DIR], server_port=port)

    return 0


def set_argparse():
    parser = argparse.ArgumentParser(
        description="Gradio based video recorder.")
    parser.add_argument("--annot-dir", default="./labels")
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
