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
Resize the guidance based on cropped vs resized image.

Version 0: The original deepedit implementation.


'''


class ResizeGuidanceMultipleLabelDeepEditd(Transform):
    """
    Resize the guidance based on cropped vs resized image.

    """

    def __init__(self, guidance: str, ref_image: str) -> None:
        self.guidance = guidance
        self.ref_image = ref_image
        self.version_param = version_param 

        self.supported_versions = ['0']

        assert self.version_param in self.supported_versions 



    def __call__(self, data):
        d = dict(data)

        if self.version_param == '0':

            # Assume channel is first and depth is last CHWD
            current_shape = d[self.ref_image].shape[1:]

            meta_dict_key = "image_meta_dict"
            # extract affine matrix from metadata
            if isinstance(d[self.ref_image], MetaTensor):
                meta_dict = d[self.ref_image].meta  # type: ignore
            elif meta_dict_key in d:
                meta_dict = d[meta_dict_key]
            else:
                raise ValueError(
                    f"{meta_dict_key} is not found. Please check whether it is the correct the image meta key."
                )

            original_shape = meta_dict["spatial_shape"]

            factor = np.divide(current_shape, original_shape)
            all_guidances = {}
            for key_label in d[self.guidance].keys():
                guidance = (
                    np.multiply(d[self.guidance][key_label], factor).astype(int).tolist()
                    if len(d[self.guidance][key_label])
                    else []
                )
                all_guidances[key_label] = guidance
                logger.info(f'Resized {key_label} guidance is {guidance}')
            
            d[self.guidance] = all_guidances
            return d