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
Extract the meta dictionary from the original state of the extracted image.


Version 1: The DeepEdit++ v1.1 implementation. 

'''

class ExtractMetad(MapTransform):
    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False,
                version_param: str = '1'
    ):
        """
        Extracting the meta information from the original state of the image. 

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            version_param: The version of this transform that is being used
        """
        super().__init__(keys, allow_missing_keys)

        self.version_param = version_param 
        self.supported_versions = ['1']

        assert self.version_param in self.supported_versions

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        if self.version_param == '1':

            for key in self.key_iterator(d):
                image = d[key]
                if isinstance(d[key], MetaTensor):
                    d["saved_meta"] = image.meta
            return d 