#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tracking: Tracking functions.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import time

# Third party's modules
import cv2

import mediapipe as mp

import numpy as np

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"


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

    if result.left_hand_landmarks is not None and len(result.left_hand_landmarks) > 0:
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.left_hand_landmarks])
        lhand_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    if result.right_hand_landmarks is not None and len(result.right_hand_landmarks) > 0:
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.right_hand_landmarks])
        rhand_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    if result.face_landmarks is not None and len(result.face_landmarks) > 0:
        _lmarks = np.array([
            [lmark.x, lmark.y, lmark.z, _binary_conf(lmark)] for lmark
            in result.face_landmarks])
        face_lmarks[0, 0, :_lmarks.shape[0], :_lmarks.shape[1]] = _lmarks
    return body_lmarks, lhand_lmarks, rhand_lmarks, face_lmarks


def track_holistic(capture, tracker):
    # Start tracking.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    landmarks = None
    while True:
        status, image = capture.read()
        if status is False:
            break

        # Tracking.
        # The OpenCV's default format is BGR, but MediaPipe requires RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int(time.time() * 1000)
        result = tracker.detect_for_video(mp_image, timestamp_ms)
        
        body, lhand, rhand, face = _ext_holistic_landmarks(result)

        landmark = np.concatenate([body, lhand, rhand, face], axis=2)
        if landmarks is None:
            landmarks = landmark
        else:
            landmarks = np.concatenate([landmarks, landmark], axis=1)
    capture.release()
    return landmarks


def _ext_pose_landmarks(result, num_detect_poses):
    # `[P, T, J, C]`
    body_lmarks = np.zeros([num_detect_poses, 1, 33, 4])
    lhand_lmarks = np.zeros([num_detect_poses, 1, 21, 4])
    rhand_lmarks = np.zeros([num_detect_poses, 1, 21, 4])
    face_lmarks = np.zeros([num_detect_poses, 1, 478, 4])

    for i, landmarks in enumerate(result.pose_landmarks):
        temp = np.array([
            [lmark.x, lmark.y, lmark.z, lmark.visibility] for lmark
            in landmarks])
        body_lmarks[i, 0, :temp.shape[0]] = temp
    return body_lmarks, lhand_lmarks, rhand_lmarks, face_lmarks


def track_pose(capture, tracker, num_detect_poses=2):
    # Start tracking.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    landmarks = None
    while True:
        status, image = capture.read()
        if status is False:
            break

        # Tracking.
        # The OpenCV's default format is BGR, but MediaPipe requires RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int(time.time() * 1000)
        result = tracker.detect_for_video(mp_image, timestamp_ms)
        
        body, lhand, rhand, face = _ext_pose_landmarks(
            result, num_detect_poses)

        landmark = np.concatenate([body, lhand, rhand, face], axis=2)
        if landmarks is None:
            landmarks = landmark
        else:
            landmarks = np.concatenate([landmarks, landmark], axis=1)
    capture.release()
    return landmarks


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
