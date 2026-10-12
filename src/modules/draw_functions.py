#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""draw_functions: Define drawing functions.
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

# Third party's modules
import cv2

import numpy as np

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


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


def draw_tracking(trackdata,
                  videopath,
                  outpath,
                  face_connections,
                  lhand_connections,
                  pose_connections,
                  rhand_connections):
    # Load video.
    capture = cv2.VideoCapture(videopath)
    if capture.isOpened() is False:
        if os.path.isfile(videopath) is False:
            raise FileNotFoundError(f"Can not find {videopath}.")
        else:
            raise RuntimeError(f"VideoCapture failed to open {videopath}.",
                               "cv2 may not link to ffmpeg or codec may be"
                               " unknown.",
                               "Please check installation of OpenCV.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    trackdata[np.isnan(trackdata)] = 0
    # If base width is smaller than 10, the trackdata should be normalized.
    # So, we project them into image plane.
    if np.abs(trackdata[:, :, :, 0].max() - trackdata[:, :, :, 0].min()) < 10:
        trackdata[:, :, :, 0] *= width
        trackdata[:, :, :, 1] *= height
    xmin_orig = trackdata[:, :, :, 0].min()
    ymin_orig = trackdata[:, :, :, 1].min()
    xmax_orig = trackdata[:, :, :, 0].max()
    ymax_orig = trackdata[:, :, :, 1].max()
    xmin_proj = int(xmin_orig)
    ymin_proj = int(ymin_orig)
    xmax_proj = int(xmax_orig)
    ymax_proj = int(ymax_orig)

    # Add offset.
    offset_x = 0
    offset_y = 0
    if xmin_proj < 0:
        offset_x = -xmin_proj
        trackdata[:, :, :, 0] += offset_x
    if ymin_proj < 0:
        offset_y = -ymin_proj
        trackdata[:, :, :, 1] += offset_y

    ywin = int(max(ymax_proj + offset_y, height + offset_y))
    xwin = int(max(xmax_proj + offset_x, width + offset_x))

    print("Window size:", xwin, ywin)
    print("Offsets:", offset_x, offset_y)

    writer = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (xwin, ywin))
    if writer.isOpened() is False:
        print("VideoWriter failed to open.")
        if writer is not None:
            writer.release()
        return

    frame_count = 0
    instances = trackdata.shape[0]
    while True:
        status, image = capture.read()
        if status is False:
            break

        # Draw
        draw = np.full([int(ywin), int(xwin), 3], 64, dtype=np.uint8)
        draw[offset_y: offset_y+height, offset_x: offset_x+width, :] = image

        for i in range(instances):
            frame = trackdata[i, frame_count]

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
        if frame_count >= trackdata.shape[1]:
            break
    writer.release()
    capture.release()


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
