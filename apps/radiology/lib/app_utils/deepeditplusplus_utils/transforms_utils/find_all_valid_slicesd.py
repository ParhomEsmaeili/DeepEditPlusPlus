from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Hashable, Mapping, Sequence, Sized

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import ScaleIntensityRange, ScaleIntensityRangePercentiles, ScaleIntensity, NormalizeIntensity, ClipIntensityPercentiles

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


'''
Finding the valid slices in a ground truth label (3D image)

Version 0: The original deepedit implementation.


'''

class FindAllValidSlicesDeepEditd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.

    Args:
        sids: key to store slices indices having valid label map.
        version_param: param which controls which version of the transform is being used.
    """

    def __init__(self, 
                keys: KeysCollection, 
                sids: Hashable = "sids", 
                allow_missing_keys: bool = False,
                version_param: str = '0'):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids
        self.version_param = version_param

        self.supported_versions = ['0']

        assert self.version_param in self.supported_versions, "The version of the transform being used is not yet supported!"


    def _apply(self, label, d):

        if self.version_param == "0":

            sids = {}
            for key_label in d["label_names"].keys():
                l_ids = []
                for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                    if d["label_names"][key_label] in label[0][..., sid]:
                        l_ids.append(sid)
                sids[key_label] = l_ids
            return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        
        if self.version_param == "0":
            for key in self.key_iterator(d):
                if key == "label":
                    label = d[key]
                    if label.shape[0] != 1:
                        raise ValueError("Only supports single channel labels!")

                    if len(label.shape) != 4:  # only for 3D
                        raise ValueError("Only supports label with shape CHWD!")

                    sids = self._apply(label, d)
                    if sids is not None and len(sids.keys()):
                        d[self.sids] = sids
                    return d
                else:
                    print("This transform only applies to label key")
            return d