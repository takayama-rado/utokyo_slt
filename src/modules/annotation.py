#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""annotation: Functions/Classes for annotation process.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from typing import (
    Any,
    Dict)

# Third party's modules


# Local modules
from .utils import get_video_infos

# Execution settings
VERSION = "%(prog)s dev"


def make_annotation(videopath: str,
                    mappings: Dict[str, str]) -> Dict[str, Any]:
    """Make annotation data.

    # Args:
      - videopath: The path to video file.
      - mappings: The mapping between video ids and glosses.

    # Returns
      - annotation: The annotation data.
    """
    video_info = get_video_infos(videopath)
    target_video = os.path.basename(videopath)
    video_id = target_video.split(".")[0]
    gloss = mappings[video_id]
    annotation = {
        "filename": video_id + ".json",
        "targetVideo": target_video,
        "videoWidth": video_info["width"],
        "videoHeight": video_info["height"],
        "videoDuration": video_info["duration"],
        "items": [
            {"start": 0.0,
             "end": video_info["duration"],
             "content": gloss
             }
        ]}
    return annotation


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
