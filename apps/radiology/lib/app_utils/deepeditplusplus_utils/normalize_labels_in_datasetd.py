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

logger = logging.getLogger(__name__)


'''

Normalise class labels implementation: This transform is intended to modify the class labels, and the values in the corresponding images (discrete) so that
they are normalized in a 0 - N_class manner with no skips. Background class will always be denoted as ZERO.

This should already have been pre-implemented in pre-processing anyways... the values should always increase by 1 from the first non-background class (background goes at the end)


Version 0: The original deepedit implementation.


'''

class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(
        self, 
        keys: KeysCollection, 
        label_names: dict[str, int] | None = None, 
        allow_missing_keys: bool = False,
        version_param: str = "0"
    ):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
            version_param: The version which we are using from this class of transform
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names or {}
        self.version_param = version_param 

        self.supported_version_params = ['0']

        assert self.version_param in self.supported_version_params, "Cannot use this class of transform as the version is not yet supported"


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        
        d: dict = dict(data)
        
        if self.version_param == '0':

            for key in self.key_iterator(d):
                # Dictionary containing new label numbers
                new_label_names = {}
                label = np.zeros(d[key].shape)
                # Making sure the range values and number of labels are the same
                for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                    if key_label != "background":
                        new_label_names[key_label] = idx
                        label[d[key] == val_label] = idx
                    if key_label == "background":
                        new_label_names["background"] = 0

                d["label_names"] = new_label_names
                if isinstance(d[key], MetaTensor):
                    d[key].array = label
                else:
                    d[key] = label
        
            return d