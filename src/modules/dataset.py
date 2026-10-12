#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""dataset: Dataset definitions and manipulation functions.
-------------------------------------------------------------------------------


Copyright (c) 2025 N.Takayama@Takayama R&D office <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import json
import os
import sys
from functools import partial

# Third party's modules
import h5py

import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    DataLoader)

from torchvision.transforms import Compose


# Local modules
from modules.transforms import (
    ToTensor
)
from modules.utils import print_exception

# Execution settings
VERSION = "%(prog)s dev"


class HDF5Dataset(Dataset):
    def __init__(self,
                 hdf5files,
                 load_into_ram=False,
                 convert_to_channel_first=False,
                 pre_transforms=None,
                 transforms=None):
        self.convert_to_channel_first = convert_to_channel_first
        self.pre_transforms = pre_transforms
        self.load_into_ram = load_into_ram
        data_info = []
        # Load file pointers.
        for fin in hdf5files:
            swap = 1 if "_swap" in fin.name else 0
            # filename should be [pid].hdf5 or [pid]_swap.hdf5
            pid = int(fin.stem.split("_")[0])
            with h5py.File(fin.resolve(), "r") as fread:
                keys = list(fread.keys())
                for key in keys:
                    if load_into_ram:
                        data = {"feature": fread[key]["feature"][:],
                                "token": fread[key]["token"][:]}
                        if self.convert_to_channel_first:
                            feature = data["feature"]
                            # `[T, J, C] -> [C, T, J]`
                            feature = np.transpose(feature, [2, 0, 1])
                            data["feature"] = feature
                        if self.pre_transforms:
                            data = self.pre_transforms(data)
                    else:
                        data = None
                    data_info.append({
                        "file": fin,
                        "data_key": key,
                        "swap": swap,
                        "pid": pid,
                        "data": data})
        self.data_info = data_info

        # Check and assign transforms.
        self.transforms = self._check_transforms(transforms)

    def _check_transforms(self, transforms):
        # Check transforms.
        if transforms:
            if isinstance(transforms, Compose):
                _transforms = transforms.transforms
            else:
                _transforms = transforms
            check_to_tensor = False
            for trans in _transforms:
                if isinstance(trans, ToTensor):
                    check_to_tensor = True
                    break
            message = "Dataset should return torch.Tensor but transforms " \
                + "does not include ToTensor class."
            assert check_to_tensor, message

        if transforms is None:
            transforms = Compose([ToTensor()])
        elif not isinstance(transforms, Compose):
            transforms = Compose(transforms)
        return transforms

    def __getitem__(self, index):
        info = self.data_info[index]
        if info["data"]:
            data = info["data"]
        else:
            with h5py.File(info["file"], "r") as fread:
                data = {"feature": fread[info["data_key"]]["feature"][:],
                        "token": fread[info["data_key"]]["token"][:]}
        _data = copy.deepcopy(data)
        if self.load_into_ram is False:
            if self.convert_to_channel_first:
                feature = _data["feature"]
                # `[T, J, C] -> [C, T, J]`
                feature = np.transpose(feature, [2, 0, 1])
                _data["feature"] = feature
            if self.pre_transforms:
                _data = self.pre_transforms(_data)
        _data = self.transforms(_data)
        _data["key"] = info["data_key"]
        return _data

    def __len__(self):
        return len(self.data_info)


class WlaslDataset(Dataset):
    def __init__(self, track_dir, label_dir, label_dictionary,
                 convert_to_channel_first=False,
                 load_into_ram=False, transforms=None):

        # Load track file pointers.
        track_dir = os.path.abspath(track_dir)
        files = os.listdir(track_dir)
        files = [fin for fin in files if fin.split(".")[-1] == "npy"]
        assert len(files) > 0, f"Can not find npy files in {track_dir}."
        self.track_files = sorted(
            [os.path.abspath(os.path.join(track_dir, fin)) for fin in files])

        # Load label file pointers.
        label_dir = os.path.abspath(label_dir)
        files = os.listdir(label_dir)
        files = [fin for fin in files if fin.split(".")[-1] == "json"]
        assert len(files) > 0, f"Can not find json files in {label_dir}."
        self.label_files = sorted(
            [os.path.abspath(os.path.join(label_dir, fin)) for fin in files])

        # Check pairs.
        message = f"{len(self.track_files)}:{len(self.label_files)}"
        assert len(self.track_files) == len(self.label_files), message
        for track, label in zip(self.track_files, self.label_files):
            base_track = os.path.basename(track).split(".")[0]
            base_label = os.path.basename(label).split(".")[0]
            message = f"Invalid file pairs, {track}, {label}"
            assert base_track == base_label, message

        if load_into_ram is True:
            self.track_files = [np.load(fin) for fin in self.track_files]
            temp = []
            for fin in self.label_files:
                with open(fin, "r") as fread:
                    temp.append(json.load(fread))
            self.label_files = temp

        # Load dictionary.
        with open(label_dictionary, "r") as fread:
            self.dictionary = json.load(fread)

        self.convert_to_channel_first = convert_to_channel_first
        self.transforms = transforms

    def __len__(self):
        return len(self.track_files)

    def __getitem__(self, idx):
        if isinstance(self.track_files[idx], str):
            # Load tracking data.
            track = np.load(self.track_files[idx])
        else:
            track = self.track_files[idx]

        # Landmarks tracked by Holistic should include a single person.
        track = track[0]
        if self.convert_to_channel_first:
            # `[T, J, C] -> [C, T, J]` for consistency.
            track = np.transpose(track, [2, 0, 1])

        if isinstance(self.label_files[idx], str):
            # Load label and convert it to tokens.
            with open(self.label_files[idx], "r") as fread:
                annot = json.load(fread)
        else:
            annot = self.label_files[idx]
        items = annot["items"]
        # Extract contents.
        contents = [item["content"] for item in items]
        # Convert contents to tokens.
        tokens = [self.dictionary[cnt]["token"] for cnt in contents]
        tokens = np.array(tokens, dtype=np.int64)

        sample = {"feature": track, "tokens": tokens}
        if self.transforms is not None:
            sample = self.transforms(sample)
        else:
            new_sample = {}
            for key, val in sample.items():
                if key == "feature":
                    assert len(val.shape) == 3, f"{val.shape}"
                if val.dtype == np.float64:
                    val = val.astype(np.float32)
                val = torch.from_numpy(val)
                new_sample[key] = val
            sample = new_sample
        return sample


def merge(sequences, merged_shape, padding_val=0):
    """ Merge variable-length sequences using padding.

    # Args:
      - sequences: The variable-length sequences.
      - merged_shape: The shape of merged batch.
      - padding_val: The value of padding. The merged batch is initialized
        with this value.
    # Returns:
      - merged: The merged batch.
    """
    merged = torch.full(tuple(merged_shape),
                        padding_val,
                        dtype=sequences[0].dtype)
    if len(merged_shape) == 2:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0]] = seq
    elif len(merged_shape) == 3:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1]] = seq
    elif len(merged_shape) == 4:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1], :seq.shape[2]] = seq
    elif len(merged_shape) == 5:
        for i, seq in enumerate(sequences):
            merged[i, :seq.shape[0], :seq.shape[1],
                   :seq.shape[2], :seq.shape[3]] = seq
    return merged


def merge_padded_batch(batch,
                       num_channels=None,
                       max_length_inputs=None,
                       max_length_outputs=None,
                       token_padding_val=-1):
    """ Merge loaded sequences into a single batch with padding.

    This method convert tokens for sequence recognition into those of
    isolated words recognition. ([<start>, word, <end>] -> [word])

    # Args:
      - batch: The initial batch data. It is assumed as list of tensors.
        Each element of the list coincides to each stream.
      - max_length_inputs: The maximum length of input sequences. The temporal
        length of merged feature batch should be this value.
      - token_padding_val: The padding value for tokens.
    # Returns:
      - retval: The tuple of merged batch including,
        - padded_feature_list: The list of merged feature batch. Each element
          of the list coincides to each stream.
        - tokens_iso: The merged tokens batch.
        - feature_pad_mask: The indicator mask of padding signals.
    """
    feature_batch = [sample["feature"] for sample in batch]
    tokens_batch = [sample["tokens"] for sample in batch]

    assert len(feature_batch[0].shape) == 3, f"{feature_batch[0].shape}"

    try:
        num_channels = feature_batch[0].shape[0]
        # `[N, C, T, J]`
        num_kpts = feature_batch[0].shape[-1]
        if max_length_inputs is None:
            max_length_inputs = max([feature.shape[1]
                                    for feature in feature_batch])
        merged_shape = [len(batch), num_channels, max_length_inputs,
                        num_kpts]
        merged_feature = merge(feature_batch, merged_shape,
                               padding_val=0)

        if max_length_outputs is None:
            max_length_outputs = max([tokens.shape[0]
                                      for tokens in tokens_batch])
        merged_shape = [len(batch), max_length_outputs]
        merged_tokens = merge(tokens_batch, merged_shape,
                              padding_val=token_padding_val)

    except Exception as inst:
        print_exception(inst)
        with open("error_dump.txt", "w") as fwrite:
            message = str(inst) + "\n" \
                + str(merged_shape) + "\n" \
                + str(feature_batch) + "\n" \
                + str(tokens_batch) + "\n"
            fwrite.write(message)
        sys.exit(2)

    # Assign True if all channels are 0.
    # We first check channel axis (dim=1), and then check kpts axis (dim=-1).
    feature_pad_mask = (merged_feature == 0).all(dim=1).all(dim=-1)
    tokens_pad_mask = merged_tokens == token_padding_val
    # 0: Pad, 1: Signal.
    feature_pad_mask = torch.logical_not(feature_pad_mask)
    tokens_pad_mask = torch.logical_not(tokens_pad_mask)
    retval = {
        "feature": merged_feature,
        "tokens": merged_tokens,
        "feature_pad_mask": feature_pad_mask,
        "tokens_pad_mask": tokens_pad_mask
    }
    return retval


def make_wlasl_dataloader(track_dir,
                          label_dir,
                          label_dictionary,
                          batch_size,
                          shuffle,
                          load_into_ram=False,
                          convert_to_channel_first=True,
                          token_padding_val=-1,
                          num_workers=0,
                          transforms=None):
    collate_fn = partial(merge_padded_batch,
                         token_padding_val=token_padding_val)
    dataset = WlaslDataset(track_dir, label_dir, label_dictionary,
                           load_into_ram=load_into_ram,
                           convert_to_channel_first=convert_to_channel_first,
                           transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers)
    return dataloader


def make_h5f_dataloader(data_paths,
                        batch_size, shuffle, load_into_ram=False,
                        convert_to_channel_first=True,
                        tokens_type="isolated",
                        num_workers=0,
                        transforms=None):
    collate_fn = partial(merge_padded_batch,
                         tokens_type=tokens_type)
    dataset = HDF5Dataset(data_paths,
                          convert_to_channel_first=convert_to_channel_first,
                          do_load_into_ram=load_into_ram,
                          transforms=transforms,
                          do_random_access=not load_into_ram)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers)
    return dataloader


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
