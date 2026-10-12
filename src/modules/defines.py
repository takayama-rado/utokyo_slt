#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""defines: Common definition.
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
from modules._defines_full import (
    USE_POSE_FULL,
    USE_LHAND_FULL,
    USE_RHAND_FULL,
    USE_FACE_FULL,
    POSE_CONNECTIONS_FULL,
    LHAND_CONNECTIONS_FULL,
    RHAND_CONNECTIONS_FULL,
    FACE_CONNECTIONS_FULL,
    POSE_ORIGIN_FULL,
    POSE_UNIT1_FULL,
    POSE_UNIT2_FULL,
    LHAND_ORIGIN_FULL,
    LHAND_UNIT1_FULL,
    LHAND_UNIT2_FULL,
    RHAND_ORIGIN_FULL,
    RHAND_UNIT1_FULL,
    RHAND_UNIT2_FULL,
    FACE_ORIGIN_FULL,
    FACE_UNIT1_FULL,
    FACE_UNIT2_FULL,
    SWAPPAIR_POSE_FULL,
    SWAPPAIR_HANDS_FULL,
    SWAPPAIR_FACE_FULL
)
from modules._defines_regular import (
    USE_POSE_REGULAR,
    USE_LHAND_REGULAR,
    USE_RHAND_REGULAR,
    USE_FACE_REGULAR,
    POSE_CONNECTIONS_REGULAR,
    LHAND_CONNECTIONS_REGULAR,
    RHAND_CONNECTIONS_REGULAR,
    FACE_CONNECTIONS_REGULAR,
    POSE_ORIGIN_REGULAR,
    POSE_UNIT1_REGULAR,
    POSE_UNIT2_REGULAR,
    LHAND_ORIGIN_REGULAR,
    LHAND_UNIT1_REGULAR,
    LHAND_UNIT2_REGULAR,
    RHAND_ORIGIN_REGULAR,
    RHAND_UNIT1_REGULAR,
    RHAND_UNIT2_REGULAR,
    FACE_ORIGIN_REGULAR,
    FACE_UNIT1_REGULAR,
    FACE_UNIT2_REGULAR,
    SWAPPAIR_POSE_REGULAR,
    SWAPPAIR_HANDS_REGULAR,
    SWAPPAIR_FACE_REGULAR
)
from modules._defines_compact import (
    USE_POSE_COMPACT,
    USE_LHAND_COMPACT,
    USE_RHAND_COMPACT,
    USE_FACE_COMPACT,
    POSE_CONNECTIONS_COMPACT,
    LHAND_CONNECTIONS_COMPACT,
    RHAND_CONNECTIONS_COMPACT,
    FACE_CONNECTIONS_COMPACT,
    POSE_ORIGIN_COMPACT,
    POSE_UNIT1_COMPACT,
    POSE_UNIT2_COMPACT,
    LHAND_ORIGIN_COMPACT,
    LHAND_UNIT1_COMPACT,
    LHAND_UNIT2_COMPACT,
    RHAND_ORIGIN_COMPACT,
    RHAND_UNIT1_COMPACT,
    RHAND_UNIT2_COMPACT,
    FACE_ORIGIN_COMPACT,
    FACE_UNIT1_COMPACT,
    FACE_UNIT2_COMPACT,
    SWAPPAIR_POSE_COMPACT,
    SWAPPAIR_HANDS_COMPACT,
    SWAPPAIR_FACE_COMPACT
)


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None

# Alias for compatibility.
USE_POSE = USE_POSE_REGULAR
USE_LHAND = USE_LHAND_REGULAR
USE_RHAND = USE_RHAND_REGULAR
USE_FACE = USE_FACE_REGULAR


# ============================================================================
# Landmarks.
# ============================================================================
def get_pose_landmarks(pose_type):
    assert pose_type in ["full", "default", "regular", "compact", "unused"]
    if pose_type in ["default", "regular"]:
        pose = USE_POSE_REGULAR
    elif pose_type == "compact":
        pose = USE_POSE_COMPACT
    elif pose_type == "full":
        pose = USE_POSE_FULL
    else:
        pose = np.array([])
    return pose


def get_lhand_landmarks(lhand_type):
    assert lhand_type in ["full", "default", "regular", "compact", "unused"]
    if lhand_type in ["default", "regular"]:
        lhand = USE_LHAND_REGULAR
    elif lhand_type == "compact":
        lhand = USE_LHAND_COMPACT
    elif lhand_type == "full":
        lhand = USE_LHAND_FULL
    else:
        lhand = np.array([])
    return lhand


def get_rhand_landmarks(rhand_type):
    assert rhand_type in ["full", "default", "regular", "compact", "unused"]
    if rhand_type in ["default", "regular"]:
        rhand = USE_RHAND_REGULAR
    elif rhand_type == "compact":
        rhand = USE_RHAND_COMPACT
    elif rhand_type == "full":
        rhand = USE_RHAND_FULL
    else:
        rhand = np.array([])
    return rhand


def get_face_landmarks(face_type):
    assert face_type in ["full", "default", "regular", "compact", "unused"]
    if face_type in ["default", "regular"]:
        face = USE_FACE_REGULAR
    elif face_type == "compact":
        face = USE_FACE_COMPACT
    elif face_type == "full":
        face = USE_FACE_FULL
    else:
        face = np.array([])
    return face


def get_fullbody_landmarks(
        pose_type="default",
        lhand_type="default",
        rhand_type="default",
        face_type="default",
        do_sort=True):
    pose = get_pose_landmarks(pose_type)
    lhand = get_lhand_landmarks(lhand_type)
    rhand = get_rhand_landmarks(rhand_type)
    face = get_face_landmarks(face_type)

    use_landmarks = np.concatenate([pose, lhand, rhand, face]).astype(np.int32)
    if do_sort:
        use_landmarks = np.sort(use_landmarks)
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks


def get_pose_hand_landmarks(
        pose_type="default",
        lhand_type="default",
        rhand_type="default",
        do_sort=True):
    return get_fullbody_landmarks(
        pose_type=pose_type, lhand_type=lhand_type, rhand_type=rhand_type,
        face_type="unused",
        do_sort=do_sort)


def get_hand_landmarks(
        lhand_type="default",
        rhand_type="default",
        do_sort=True):
    assert lhand_type in ["full", "default", "regular", "compact"]
    assert rhand_type in ["full", "default", "regular", "compact"]

    pose = np.arange(15, 23)
    if lhand_type in ["default", "regular"]:
        lhand = USE_LHAND_REGULAR
    elif lhand_type == "compact":
        lhand = USE_LHAND_COMPACT
    elif lhand_type == "full":
        lhand = USE_LHAND_FULL
    if rhand_type in ["default", "regular"]:
        rhand = USE_RHAND_REGULAR
    elif rhand_type == "compact":
        rhand = USE_RHAND_COMPACT
    elif rhand_type == "full":
        rhand = USE_RHAND_FULL
    use_landmarks = np.concatenate([pose, lhand, rhand])
    if do_sort:
        use_landmarks = np.sort(use_landmarks)
    use_landmarks_filtered = np.arange(len(use_landmarks))
    return use_landmarks_filtered, use_landmarks


# ============================================================================
# Landmarks for calculation.
# ============================================================================
# Origin.
def get_pose_origin(pose_type):
    assert pose_type in ["full", "default", "regular", "compact"]
    if pose_type in ["default", "regular"]:
        origin = POSE_ORIGIN_REGULAR
    elif pose_type == "compact":
        origin = POSE_ORIGIN_COMPACT
    elif pose_type == "full":
        origin = POSE_ORIGIN_FULL
    return origin


def get_lhand_origin(lhand_type):
    assert lhand_type in ["full", "default", "regular", "compact"]
    if lhand_type in ["default", "regular"]:
        origin = LHAND_ORIGIN_REGULAR
    elif lhand_type == "compact":
        origin = LHAND_ORIGIN_COMPACT
    elif lhand_type == "full":
        origin = LHAND_ORIGIN_FULL
    return origin


def get_rhand_origin(rhand_type):
    assert rhand_type in ["full", "default", "regular", "compact"]
    if rhand_type in ["default", "regular"]:
        origin = RHAND_ORIGIN_REGULAR
    elif rhand_type == "compact":
        origin = RHAND_ORIGIN_COMPACT
    elif rhand_type == "full":
        origin = RHAND_ORIGIN_FULL
    return origin


def get_face_origin(face_type):
    assert face_type in ["full", "default", "regular", "compact"]
    if face_type in ["default", "regular"]:
        origin = FACE_ORIGIN_REGULAR
    elif face_type == "compact":
        origin = FACE_ORIGIN_COMPACT
    elif face_type == "full":
        origin = FACE_ORIGIN_FULL
    return origin


# Unit1.
def get_pose_unit1(pose_type):
    assert pose_type in ["full", "default", "regular", "compact"]
    if pose_type in ["default", "regular"]:
        unit1 = POSE_UNIT1_REGULAR
    elif pose_type == "compact":
        unit1 = POSE_UNIT1_COMPACT
    elif pose_type == "full":
        unit1 = POSE_UNIT1_FULL
    return unit1


def get_lhand_unit1(lhand_type):
    assert lhand_type in ["full", "default", "regular", "compact"]
    if lhand_type in ["default", "regular"]:
        unit1 = LHAND_UNIT1_REGULAR
    elif lhand_type == "compact":
        unit1 = LHAND_UNIT1_COMPACT
    elif lhand_type == "full":
        unit1 = LHAND_UNIT1_FULL
    return unit1


def get_rhand_unit1(rhand_type):
    assert rhand_type in ["full", "default", "regular", "compact"]
    if rhand_type in ["default", "regular"]:
        unit1 = RHAND_UNIT1_REGULAR
    elif rhand_type == "compact":
        unit1 = RHAND_UNIT1_COMPACT
    elif rhand_type == "full":
        unit1 = RHAND_UNIT1_FULL
    return unit1


def get_face_unit1(face_type):
    assert face_type in ["full", "default", "regular", "compact"]
    if face_type in ["default", "regular"]:
        unit1 = FACE_UNIT1_REGULAR
    elif face_type == "compact":
        unit1 = FACE_UNIT1_COMPACT
    elif face_type == "full":
        unit1 = FACE_UNIT1_FULL
    return unit1


# Unit2.
def get_pose_unit2(pose_type):
    assert pose_type in ["full", "default", "regular", "compact"]
    if pose_type in ["default", "regular"]:
        unit2 = POSE_UNIT2_REGULAR
    elif pose_type == "compact":
        unit2 = POSE_UNIT2_COMPACT
    elif pose_type == "full":
        unit2 = POSE_UNIT2_FULL
    return unit2


def get_lhand_unit2(lhand_type):
    assert lhand_type in ["full", "default", "regular", "compact"]
    if lhand_type in ["default", "regular"]:
        unit2 = LHAND_UNIT2_REGULAR
    elif lhand_type == "compact":
        unit2 = LHAND_UNIT2_COMPACT
    elif lhand_type == "full":
        unit2 = LHAND_UNIT2_FULL
    return unit2


def get_rhand_unit2(rhand_type):
    assert rhand_type in ["full", "default", "regular", "compact"]
    if rhand_type in ["default", "regular"]:
        unit2 = RHAND_UNIT2_REGULAR
    elif rhand_type == "compact":
        unit2 = RHAND_UNIT2_COMPACT
    elif rhand_type == "full":
        unit2 = RHAND_UNIT2_FULL
    return unit2


def get_face_unit2(face_type):
    assert face_type in ["full", "default", "regular", "compact"]
    if face_type in ["default", "regular"]:
        unit2 = FACE_UNIT2_REGULAR
    elif face_type == "compact":
        unit2 = FACE_UNIT2_COMPACT
    elif face_type == "full":
        unit2 = FACE_UNIT2_FULL
    return unit2


MAPPINGS_CALC_LMARKS = {
    "pose": {
        "origin": get_pose_origin,
        "unit1": get_pose_unit1,
        "unit2": get_pose_unit2
    },
    "lhand": {
        "origin": get_lhand_origin,
        "unit1": get_lhand_unit1,
        "unit2": get_lhand_unit2
    },
    "rhand": {
        "origin": get_rhand_origin,
        "unit1": get_rhand_unit1,
        "unit2": get_rhand_unit2
    },
    "face": {
        "origin": get_face_origin,
        "unit1": get_face_unit1,
        "unit2": get_face_unit2
    },
}


def get_calculation_landmarks(parts_name, parts_type, unit_type):
    assert parts_name in ["pose", "lhand", "rhand", "face"], parts_name
    assert parts_type in ["full", "default", "regular", "compact"], parts_type
    assert unit_type in ["origin", "unit1", "unit2"], unit_type
    return MAPPINGS_CALC_LMARKS[parts_name][unit_type](parts_type)


# ============================================================================
# Swap pairs.
# ============================================================================
def get_pose_swappairs(pose_type):
    assert pose_type in ["full", "default", "regular", "compact"]
    if pose_type in ["default", "regular"]:
        swap_pairs = SWAPPAIR_POSE_REGULAR
    elif pose_type == "compact":
        swap_pairs = SWAPPAIR_POSE_COMPACT
    elif pose_type == "full":
        swap_pairs = SWAPPAIR_POSE_FULL
    return swap_pairs


def get_hands_swappairs(hand_type):
    assert hand_type in ["full", "default", "regular", "compact"]
    if hand_type in ["default", "regular"]:
        swap_pairs = SWAPPAIR_HANDS_REGULAR
    elif hand_type == "compact":
        swap_pairs = SWAPPAIR_HANDS_COMPACT
    elif hand_type == "full":
        swap_pairs = SWAPPAIR_HANDS_FULL
    return swap_pairs


def get_face_swappairs(face_type):
    assert face_type in ["full", "default", "regular", "compact"]
    if face_type in ["default", "regular"]:
        swap_pairs = SWAPPAIR_FACE_REGULAR
    elif face_type == "compact":
        swap_pairs = SWAPPAIR_FACE_COMPACT
    elif face_type == "full":
        swap_pairs = SWAPPAIR_FACE_FULL
    return swap_pairs


def get_fullbody_swappairs(
        pose_type="default",
        lhand_type="default",
        rhand_type="default",
        face_type="default",
        do_sort=True):
    _, use_landmarks = get_fullbody_landmarks(
        pose_type, lhand_type, rhand_type, face_type, do_sort)
    assert lhand_type == rhand_type, f"{lhand_type}:{rhand_type}"
    swappair = np.concatenate(
        [get_pose_swappairs(pose_type),
         get_hands_swappairs(lhand_type),
         get_face_swappairs(rhand_type)], axis=0)
    swappair_filtered = np.array([
        [pt[0], pt[1]] for pt in swappair
        if pt[0] in use_landmarks and pt[1] in use_landmarks])
    swappair_filtered = np.concatenate(
        [np.where(pt == use_landmarks)[0]
         for pt in swappair_filtered.flatten()])
    swappair_filtered = swappair_filtered.reshape([-1, 2])
    return swappair_filtered, swappair


# ============================================================================
# Connections.
# ============================================================================
def get_pose_connections(pose_type):
    assert pose_type in ["full", "default", "regular", "compact"]
    if pose_type in ["default", "regular"]:
        connections = POSE_CONNECTIONS_REGULAR
    elif pose_type == "compact":
        connections = POSE_CONNECTIONS_COMPACT
    elif pose_type == "full":
        connections = POSE_CONNECTIONS_FULL
    return connections


def get_lhand_connections(lhand_type):
    assert lhand_type in ["full", "default", "regular", "compact"]
    if lhand_type in ["default", "regular"]:
        connections = LHAND_CONNECTIONS_REGULAR
    elif lhand_type == "compact":
        connections = LHAND_CONNECTIONS_COMPACT
    elif lhand_type == "full":
        connections = LHAND_CONNECTIONS_FULL
    return connections


def get_rhand_connections(rhand_type):
    assert rhand_type in ["full", "default", "regular", "compact"]
    if rhand_type in ["default", "regular"]:
        connections = RHAND_CONNECTIONS_REGULAR
    elif rhand_type == "compact":
        connections = RHAND_CONNECTIONS_COMPACT
    elif rhand_type == "full":
        connections = RHAND_CONNECTIONS_FULL
    return connections


def get_face_connections(face_type):
    assert face_type in ["full", "default", "regular", "compact"]
    if face_type in ["default", "regular"]:
        connections = FACE_CONNECTIONS_REGULAR
    elif face_type == "compact":
        connections = FACE_CONNECTIONS_COMPACT
    elif face_type == "full":
        connections = FACE_CONNECTIONS_FULL
    return connections


def get_parts_connections(use_landmarks, partsname, partstype="default"):
    if partsname == "pose":
        connections = get_pose_connections(partstype)
    elif partsname == "lhand":
        connections = get_lhand_connections(partstype)
    elif partsname == "rhand":
        connections = get_rhand_connections(partstype)
    elif partsname == "face":
        connections = get_face_connections(partstype)
    else:
        raise ValueError(f"Unknown parts name: {partsname}.")
    connections = np.array(
        [[pair[0], pair[1]] for pair in connections
         if pair[0] in use_landmarks and pair[1] in use_landmarks])
    connections_sorted = np.array([[np.where(pair[0] == use_landmarks)[0][0],
                                    np.where(pair[1] == use_landmarks)[0][0]]
                                   for pair in connections])
    return connections, connections_sorted


# ============================================================================
# Misc.
# ============================================================================
# Create data columns.
def create_data_columns(indices, parts_name, include_z=False):
    assert parts_name in ["face", "left_hand",
                          "right_hand", "pose"], f"{parts_name}"
    columns = []
    for index in indices:
        col = [f"x_{parts_name}_{index}", f"y_{parts_name}_{index}"]
        if include_z:
            col += [f"z_{parts_name}_{index}"]
        columns += col
    return columns


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
