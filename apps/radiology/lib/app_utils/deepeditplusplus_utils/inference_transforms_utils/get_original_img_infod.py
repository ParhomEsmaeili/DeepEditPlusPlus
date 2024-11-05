
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

class GetOriginalInformationd(MapTransform):
    def __init__(self, keys: KeysCollection, version_param: str = '0', allow_missing_keys: bool = False):
        """
        Get information from original image

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d["original_size"] = d[key].shape[-3], d[key].shape[-2], d[key].shape[-1]
        return d