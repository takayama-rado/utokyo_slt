#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""_defines_regular: Landmark definition (Regular).
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# Third party's modules
import numpy as np

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None

# ============================================================================
# Landmarks.
# ============================================================================
USE_POSE_REGULAR = np.arange(0, 23)
USE_LHAND_REGULAR = np.arange(33, 33+21)
USE_RHAND_REGULAR = np.arange(33+21, 33+21+21)

USE_LIP_REGULAR = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
USE_NOSE_REGULAR = [98, 1, 327, 2]
USE_REYE_REGULAR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    246, 161, 160, 159, 158, 157, 173]
USE_LEYE_REGULAR = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                    466, 388, 387, 386, 385, 384, 398]
USE_FACE_REGULAR = np.array(
    USE_LIP_REGULAR + USE_NOSE_REGULAR + USE_REYE_REGULAR + USE_LEYE_REGULAR) \
    + 33 + 21 + 21

# ============================================================================
# Landmarks for calculation.
# ============================================================================
POSE_ORIGIN_REGULAR = [11, 12]  # Neck (between shoulder).
POSE_UNIT1_REGULAR = [11]  # LShoulder.
POSE_UNIT2_REGULAR = [12]  # RShoulder.

LHAND_ORIGIN_REGULAR = [0, 2, 5, 9, 13, 17]  # Center of parm.
LHAND_UNIT1_REGULAR = [0]  # Wrist.
LHAND_UNIT2_REGULAR = [2, 5, 9, 13, 17]  # Center of MCP.

RHAND_ORIGIN_REGULAR = [0, 2, 5, 9, 13, 17]  # Center of parm.
RHAND_UNIT1_REGULAR = [0]  # Wrist.
RHAND_UNIT2_REGULAR = [2, 5, 9, 13, 17]  # Center of MCP.

FACE_ORIGIN_REGULAR = [1, 2]  # Center of nose.
FACE_UNIT1_REGULAR = [7]  # Outer corner of right eye.
FACE_UNIT2_REGULAR = [42]  # Outer corner of left eye.

# ============================================================================
# Swap pairs.
# ============================================================================
SWAPPAIR_POSE_REGULAR = np.array([
    [1, 4], [2, 5], [3, 6], [7, 8], [9, 10],
    [11, 12], [13, 14], [15, 16], [17, 18], [19, 20],
    [21, 22]])
SWAPPAIR_HANDS_REGULAR = np.array(
    [[pl, pr] for pl, pr
     in zip(USE_LHAND_REGULAR, USE_RHAND_REGULAR)])

# Face swap pairs.
SWAPPAIR_LIP_REGULAR = np.array([
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
SWAPPAIR_NOSE_REGULAR = np.array([[98, 327]])
SWAPPAIR_EYE_REGULAR = np.array([
    # Upper line.
    [173, 398], [157, 384], [158, 385], [159, 386], [160, 387],
    [161, 388], [246, 466],
    # Both corner.
    [133, 362], [33, 263],
    # Lower line.
    [155, 382], [154, 381], [153, 380], [145, 374], [144, 373],
    [163, 390], [7, 249]])

SWAPPAIR_FACE_REGULAR = np.concatenate(
    [SWAPPAIR_LIP_REGULAR,
     SWAPPAIR_NOSE_REGULAR,
     SWAPPAIR_EYE_REGULAR], axis=0) + 33 + 21 + 21

# ============================================================================
# Connections.
# ============================================================================
POSE_CONNECTIONS_REGULAR = np.array([
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
    [5, 6], [6, 8], [9, 10], [0, 9], [0, 10], [0, 11], [0, 12],
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19],
    [15, 21], [17, 19], [12, 14], [14, 16], [16, 18],
    [16, 20], [16, 22], [18, 20]])
HAND_CONNECTIONS_REGULAR = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]])
LHAND_CONNECTIONS_REGULAR = HAND_CONNECTIONS_REGULAR + 33
RHAND_CONNECTIONS_REGULAR = HAND_CONNECTIONS_REGULAR + 21 + 33

LIP_CONNECTIONS_REGULAR = np.array([
    # Outer line.
    [0, 37], [37, 39], [39, 40], [40, 185], [185, 61],
    [61, 146], [146, 91], [91, 181], [181, 84], [84, 17],
    [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
    [291, 409], [409, 270], [270, 269], [269, 267], [267, 0],
    # Inner line.
    [13, 82], [82, 81], [81, 80], [80, 191], [191, 78],
    [78, 95], [95, 88], [88, 178], [178, 87], [87, 14],
    [317, 402], [402, 318], [318, 324], [324, 308], [308, 415],
    [415, 310], [310, 311], [311, 312], [312, 13],
    # Outer to inner.
    [61, 78], [291, 308]
])
NOSE_CONNECTIONS_REGULAR = np.array([
    [1, 98], [98, 2], [2, 327], [327, 1]
])
REYE_CONNECTIONS_REGULAR = np.array([
    [133, 173], [173, 157], [157, 158], [158, 159], [159, 160],
    [161, 246], [246, 33], [33, 7], [7, 163], [163, 144],
    [144, 145], [145, 153], [153, 154], [154, 155], [155, 133]
])
LEYE_CONNECTIONS_REGULAR = np.array([
    [362, 398], [398, 384], [384, 385], [385, 386], [386, 387],
    [387, 388], [388, 466], [466, 263], [263, 249], [249, 390],
    [390, 373], [373, 374], [374, 380], [380, 381], [381, 382],
    [382, 362]
])
FACE_INNER_CONNECTIONS_REGULAR = np.array([
    [1, 133], [1, 362], [2, 0]
])
FACE_CONNECTIONS_REGULAR = np.concatenate(
    [LIP_CONNECTIONS_REGULAR,
     NOSE_CONNECTIONS_REGULAR,
     REYE_CONNECTIONS_REGULAR,
     LEYE_CONNECTIONS_REGULAR,
     FACE_INNER_CONNECTIONS_REGULAR], axis=0) + 33 + 21 + 21
