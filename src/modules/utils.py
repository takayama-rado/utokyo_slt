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

import subprocess
import sys
import random
import traceback
from typing import (
    Any,
    Dict,
    Union)

# Third party's modules
import cv2

import numpy as np

import torch

from torch import nn

# Local modules


# Execution settings
VERSION = "%(prog)s dev"


def print_exception(ex: Exception,
                    message: str = "",
                    add_separator: bool = True) -> None:
    """Print detail of exception and traceback.

    # Args:
        - message: Any message to output with exception.
        - add_separator: If True, the message separator "---" is added.

    """
    print(type(ex))
    print(ex.args)
    print(ex)
    print(message)
    traceback.print_exc(file=sys.stdout)
    if add_separator is True:
        print("-" * 60)


def get_video_infos(path: str) -> Dict[str, Any]:
    """Get video information.

    # Args:
      - path: The path of video file.

    # Returns:
      - info: The directory containing video information.
        - height: The video height.
        - width: The video width.
        - frame_count: The number of frames.
        - fps: The frame per second.
        - duration: The video duration.
    """
    capture = cv2.VideoCapture(path)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    capture.release()
    info = {
        "height": height,
        "width": width,
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration}
    return info


def callsubproc(command: str) -> Union[int, None]:
    """Call sub process.

    # Args:
        - command: The sub process and its arguments.

    # Returns:
        - ret: Return code of sub process.
    """
    print("-" * 80)
    print(command, "\n")
    proc = subprocess.Popen(command, shell=True, cwd="./",
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            close_fds=sys.platform != "win32")

    stdout_err = proc.stdout
    while stdout_err is not None:
        line = str(stdout_err.readline(), encoding="utf-8")
        if not line:
            break
        print(line.rstrip())
    print("-" * 80)
    ret = proc.wait()
    return ret


class Zero(nn.Module):
    """Place holder layer to return zero vector.
    """
    # pylint: disable=unused-argument, no-self-use

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return 0


class Identity(nn.Module):
    """Place holder layer to return identity vector.
    """
    # pylint: disable=unused-argument, no-self-use

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return feature


def save_checkpoint(ckpt_path, model,
                    epoch=0,
                    optimizer=None,
                    scheduler=None):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {}
    checkpoint = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),
        "cuda_random_all": torch.cuda.get_rng_state_all(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)


def load_checkpoint(ckpt_path, model,
                    optimizer=None,
                    scheduler=None):
    checkpoint = torch.load(ckpt_path, weights_only=False)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    epoch = checkpoint["epoch"]
    random.setstate(checkpoint["random"])
    np.random.set_state(checkpoint["np_random"])
    torch.set_rng_state(checkpoint["torch"])
    torch.random.set_rng_state(checkpoint["torch_random"])
    try:
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
    except Exception as inst:
        print("Info: The original model is trained with CPU. "
              "If it intended, you can ignore bellow exception.")
        print_exception(inst)
        pass

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, epoch, optimizer, scheduler


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
