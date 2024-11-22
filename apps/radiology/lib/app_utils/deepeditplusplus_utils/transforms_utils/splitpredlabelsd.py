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

Split predictions and labels for individual evaluation.

Version 0: The original deepedit implementation.

'''

# class SplitPredsLabeld(MapTransform):
#     """
#     Split preds and labels for individual evaluation

#     """
#     def __init__(self,
#                 keys: KeysCollection,
#                 allow_missing_keys: bool = False,
#                 version_param : str = '0'):
        
#         super().__init__(keys, allow_missing_keys)

#         self.version_param = version_param 

#         self.supported_versions = ['0']

#         assert self.version_param in self.supported_versions 

#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
#         d: dict = dict(data)
        
#         if self.version_param == ['0']:
#             for key in self.key_iterator(d):
#                 if key == "pred":
#                     for idx, (key_label, _) in enumerate(d["label_names"].items()):
#                         if key_label != "background":
#                             d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
#                             d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
#                 elif key != "pred":
#                     logger.info("This is only for pred key")
#             print(d)
#             return d

class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    Version param=0: For splitting a prediction which is one hot encoded for a NHWD shape where N = number of pred channels (for decollated batch)

    """
    def __init__(self,
                keys: KeysCollection,
                allow_missing_keys: bool = False,
                version_param : str = '0'):
        
        super().__init__(keys, allow_missing_keys)

        self.version_param = version_param 

        self.supported_versions = ['0']#, '1']

        assert self.version_param in self.supported_versions 

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        

        d: dict = dict(data)

        if self.version_param == '0':
            for key in self.key_iterator(d):
                if key == "pred":
                    for idx, (key_label, _) in enumerate(d["label_names"].items()):
                        if key_label != "background":
                            d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                            d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
                            #Output shape is 1HWD for each class.
                elif key != "pred":
                    logger.info("This is only for pred key")
            
        # elif self.version_param == '1':

        #     for key in self.key_iterator(d):
        #         if key == "pred_output":
        #             for idx, (key_label, _) in enumerate(d["label_names"].items()):
        #                 if key_label != "background":
        #                     d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
        #                     d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]

        #         elif key != "pred":
        #             logger.info("This is only for pred key")
            
        return d