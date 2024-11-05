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

Function which maps the label config from an original config to a new one using a label mapping definition.
Function also maps the values of the input image (intended to be a ground truth or a discrete prediction) according to this label mapping (and the label codes)

Version 1: The DeepEdit++ v1.1 implementation.  (DEPRECATED, REPLACED WITH PREPROCESSING OF THE DATASET TO A NORMALIZED CONFIG LABEL!)
Version 2: The DeepEdit++ v1.1 implementation in TORCH. NOT SUPPORTED 

'''

class MappingLabelsInDatasetd(MapTransform):
    def __init__(
        self, keys: KeysCollection, 
        original_label_names: dict[str, int] | None = None, 
        label_names: dict[str, int] | None = None, 
        label_mapping: dict[str, list] | None = None, 
        allow_missing_keys: bool = False,
        version_param: str = '1'
    ):
        """
        Changing the labels from the original dataset config, to whatever is the new desired config, according to a mapping.  

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)
        self.original_label_names = original_label_names
        self.label_names = label_names
        self.label_mapping = label_mapping 
        self.version_param = version_param 

        self.supported_versions = ['1']
        assert self.version_param in self.supported_versions, 'Attempted to use the transform with an invalid version/non supported version'

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
    
        if self.version_param == '1':
    
            d: dict = dict(data)
            for key in self.key_iterator(d):
                # Dictionary containing new label numbers
                label = np.zeros(d[key].shape)

                for (key_label, val_label) in self.label_names.items():
                    #For each key label in the "new" config, extract the mapped classes from the original set to the current set
                    mapping_list = self.label_mapping[key_label]
                    #For each of the labels in the mapping list, convert the voxels with those values to what they are being mapped to
                    for key_label_original in mapping_list:
                        label[d[key] == self.original_label_names[key_label_original]] = val_label
                    
                if isinstance(d[key], MetaTensor):
                    d[key].array = label
                else:
                    d[key] = label
            return d
