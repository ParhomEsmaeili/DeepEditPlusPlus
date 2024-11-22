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
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)


class ExtractMapsd(MapTransform):
    """
    Extracts 
    Version param=0: For extracting from a deep supervision FNHWD tensor (the batch is already decollated), the feature maps which correspond to the output of the network. 
    (Not the intermediate feature maps, this is contained in the first index of the feature map channel wise axis.)

    """
    def __init__(self,
                keys: KeysCollection,
                allow_missing_keys: bool = False,
                version_param : str = '0'):
        
        super().__init__(keys, allow_missing_keys)

        self.version_param = version_param 

        self.supported_versions = ['0']

        assert self.version_param in self.supported_versions 

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        
        d: dict = dict(data)

        if self.version_param == '0':
            for key in self.key_iterator(d):
                if key == "pred":
                    d["pred"] = d[key][0]
                elif key != "pred":
                    logger.info("This is only for pred key")
            
    
        return d