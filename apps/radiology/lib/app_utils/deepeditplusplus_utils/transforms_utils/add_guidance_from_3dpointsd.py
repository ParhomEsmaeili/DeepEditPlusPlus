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

Mapping points to the dimensions/coordinates in the resolution that the backend is working in. 

Intended as a transform for performing inference.

The assumption here is that a resizing is the only operation that has been implemented with regards to changing the dimensions of the image.


Version 0: The original deepedit implementation.


'''


class AddGuidanceFromPointsDeepEditd(Transform):
    """
    Add guidance based on user clicks. ONLY WORKS FOR 3D

    We assume the input is loaded by LoadImaged and has the shape of (H, W, D) originally.
    Clicks always specify the coordinates in (H, W, D)

    Args:
        ref_image: key to reference image to fetch current and original image details.
        
        guidance: output key to store guidance.
        
        meta_keys: explicitly indicate the key of the metadata dictionary of `ref_image`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{ref_image}_{meta_key_postfix}`.
        
        meta_key_postfix: if meta_key is None, use `{ref_image}_{meta_key_postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.

        version_param: The version of the transform that is being used! 

    """

    def __init__(
        self,
        ref_image: str,
        guidance: str = "guidance",
        label_names: dict | None = None,
        meta_keys: str | None = None,
        meta_key_postfix: str = "meta_dict",
        version_param: str = '0'
    ):
        self.ref_image = ref_image
        self.guidance = guidance
        self.label_names = label_names or {}
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix
        self.version_param = version_param 

        self.supported_versions = ['0']

        assert self.version_param in self.supported_versions

    @staticmethod
    def _apply(clicks, factor):
        if self.version_param == '0':
            if len(clicks):
                guidance = np.multiply(clicks, factor).astype(int).tolist()
                return guidance
            else:
                return []

    def __call__(self, data):
        d = dict(data)

        if self.version_param == '0':

            meta_dict_key = self.meta_keys or f"{self.ref_image}_{self.meta_key_postfix}"

            if isinstance(d[self.ref_image], MetaTensor):
                meta_dict = d[self.ref_image].meta  # type: ignore
            elif meta_dict_key in d:
                meta_dict = d[meta_dict_key]
            else:
                raise ValueError(
                    f"{meta_dict_key} is not found. Please check whether it is the correct the image meta key."
                )

            if "spatial_shape" not in meta_dict:
                raise RuntimeError('Missing "spatial_shape" in meta_dict!')

            # Assume channel is first and depth is last CHWD
            original_shape = meta_dict["spatial_shape"]
            current_shape = list(d[self.ref_image].shape)[1:]

            factor = np.array(current_shape) / original_shape

            # Creating guidance for all clicks
            all_guidances = {}
            for key_label in self.label_names.keys():
                clicks = d.get(key_label, [])
                clicks = list(np.array(clicks).astype(int))
                all_guidances[key_label] = self._apply(clicks, factor)
            d[self.guidance] = all_guidances

            return d