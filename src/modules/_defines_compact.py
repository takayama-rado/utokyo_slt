#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""_defines_compact: Landmark definition (Compact).
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
USE_POSE_COMPACT = np.array([
    0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
USE_LHAND_COMPACT = np.array(
    [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]) + 33
USE_RHAND_COMPACT = np.array(
    [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]) + 33 + 21

USE_LIP_COMPACT = [61, 0, 291, 17]
USE_NOSE_COMPACT = [98, 1, 327, 2]
USE_REYE_COMPACT = [33, 159, 133, 145]
USE_LEYE_COMPACT = [362, 386, 263, 374]
USE_FACE_COMPACT = np.array(
    USE_LIP_COMPACT + USE_NOSE_COMPACT +
    USE_REYE_COMPACT + USE_LEYE_COMPACT) + 33 + 21 + 21

# ============================================================================
# Landmarks for calculation.
# ============================================================================
POSE_ORIGIN_COMPACT = [1, 2]  # Neck (between shoulder).
POSE_UNIT1_COMPACT = [1]  # LShoulder.
POSE_UNIT2_COMPACT = [2]  # RShoulder.

LHAND_ORIGIN_COMPACT = [0, 3, 5, 7, 9]  # Center of parm.
LHAND_UNIT1_COMPACT = [0]  # Wrist.
LHAND_UNIT2_COMPACT = [3, 5, 7, 9]  # Center of MCP.

RHAND_ORIGIN_COMPACT = [0, 3, 5, 7, 9]  # Center of parm.
RHAND_UNIT1_COMPACT = [0]  # Wrist.
RHAND_UNIT2_COMPACT = [3, 5, 7, 9]  # Center of MCP.

FACE_ORIGIN_COMPACT = [1, 2]  # Center of nose.
FACE_UNIT1_COMPACT = [4]  # Outer corner of right eye.
FACE_UNIT2_COMPACT = [10]  # Outer corner of left eye.

# ============================================================================
# Swap pairs.
# ============================================================================
SWAPPAIR_POSE_COMPACT = np.array([
    [11, 12], [13, 14], [15, 16], [17, 18], [19, 20],
    [21, 22]])
SWAPPAIR_HANDS_COMPACT = np.array(
    [[pl, pr] for pl, pr
     in zip(USE_LHAND_COMPACT, USE_RHAND_COMPACT)])

# Face swap pairs.
SWAPPAIR_LIP_COMPACT = np.array([[61, 291]])
SWAPPAIR_NOSE_COMPACT = np.array([[98, 327]])
SWAPPAIR_EYE_COMPACT = np.array([
    [133, 362], [33, 263]])

SWAPPAIR_FACE_COMPACT = np.concatenate(
    [SWAPPAIR_LIP_COMPACT,
     SWAPPAIR_NOSE_COMPACT,
     SWAPPAIR_EYE_COMPACT], axis=0) + 33 + 21 + 21

# ============================================================================
# Connections.
# ============================================================================
POSE_CONNECTIONS_COMPACT = np.array([
    [0, 11], [0, 12],
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [11, 12],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20]])
HAND_CONNECTIONS_COMPACT = np.array([
    [0, 2], [2, 4],
    [0, 5], [5, 8],
    [0, 9], [9, 12],
    [0, 13], [13, 16],
    [0, 17], [17, 20]])
LHAND_CONNECTIONS_COMPACT = HAND_CONNECTIONS_COMPACT + 33
RHAND_CONNECTIONS_COMPACT = HAND_CONNECTIONS_COMPACT + 21 + 33

LIP_CONNECTIONS_COMPACT = np.array([
    [0, 61], [61, 17], [17, 291], [291, 0]])
NOSE_CONNECTIONS_COMPACT = np.array([
    [1, 98], [98, 2], [2, 327], [327, 1]])
REYE_CONNECTIONS_COMPACT = np.array([
    [33, 159], [159, 133], [133, 145], [145, 33]])
LEYE_CONNECTIONS_COMPACT = np.array([
    [263, 386], [386, 362], [362, 374], [374, 263]
])
FACE_INNER_CONNECTIONS_COMPACT = np.array([
    [1, 133], [1, 362], [2, 0]
])
FACE_CONNECTIONS_COMPACT = np.concatenate(
    [LIP_CONNECTIONS_COMPACT,
     NOSE_CONNECTIONS_COMPACT,
     REYE_CONNECTIONS_COMPACT,
     LEYE_CONNECTIONS_COMPACT,
     FACE_INNER_CONNECTIONS_COMPACT], axis=0) + 33 + 21 + 21
