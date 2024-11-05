
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

Discard add guidance version information:

Version 0: The original deepedit implementation.
Version 1: The DeepEdit++ v1.1 implementation. 
Version 2: The DeepEdit++ v1.1 implementation in TORCH. 

'''
class DiscardAddGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        number_intensity_ch: int = 1,
        probability: float = 1.0,
        label_names: Sized | None = None,
        allow_missing_keys: bool = False,
        version_param: str = '1'
    ):
        """
        Discard points according to discard probability

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            number_intensity_ch: number of intensity channels
            probability: probability of discarding clicks
            version_param: The parameter which controls which version of this class and its methods we are using.
        """
        super().__init__(keys, allow_missing_keys)

        self.number_intensity_ch = number_intensity_ch
        self.discard_probability = probability
        self.label_names = label_names or []
        self.version_param = version_param 

        self.supported_version_params = ['0', '1']

        assert self.version_param in self.supported_version_params, "Cannot use this class of transform as the version is not yet supported"


    def _apply(self, image):
        
        if self.version_param == '1':

            if self.discard_probability >= 1.0 or np.random.choice([True, False], p=[self.discard_probability, 1 - self.discard_probability]):

                signal = np.zeros((len(self.label_names), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)

                if image.shape[0] == self.number_intensity_ch + 2 * len(self.label_names):
                    image[self.number_intensity_ch: self.number_intensity_ch + len(self.label_names), ...] = signal
                else:
                    image = np.concatenate([image, signal], axis=0)

        return image

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        



        if self.version_param == '1':
            for key in self.key_iterator(d):
                if key == "image":
                    # print(d[key].shape)
                    tmp_image = self._apply(d[key])
                    if isinstance(d[key], MetaTensor):
                        # print(tmp_image.shape)
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                else:
                    print("This transform only applies to the image")




        return d