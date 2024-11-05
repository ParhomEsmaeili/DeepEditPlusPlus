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
Finding the discrepancies between a prediction and a ground truth label for each class.

Version 0: The original deepedit implementation.


'''

class FindDiscrepancyRegionsDeepEditd(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
        version_param: The parameter which determines which version of the transform that is being used.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred: str = "pred",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
        version_param: str = '0'
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred = pred
        self.discrepancy = discrepancy
        self.version_param = version_param 

        self.supported_versions = ['0']

        assert self.version_param in self.supported_versions, "The version parameter selected is not currently supported by this transform."

    @staticmethod
    def disparity(label, pred):
        disparity = label - pred
        # Negative ONES mean predicted label is not part of the ground truth
        # Positive ONES mean predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
    
        if self.version_param == '0':
            return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        
        if self.version_param == '0':
            for key in self.key_iterator(d):
                if key == "label":
                    all_discrepancies = {}
                    
                    for _, (key_label, val_label) in enumerate(d["label_names"].items()):
                        if key_label != "background":
                            # Taking single label
                            label = np.copy(d[key])
                            label[label != val_label] = 0
                            # Label should be represented in 1
                            label = (label > 0.5).astype(np.float32)
                            # Taking single prediction
                            pred = np.copy(d[self.pred])
                            pred[pred != val_label] = 0
                            # Prediction should be represented in one
                            pred = (pred > 0.5).astype(np.float32)
                        else:
                            # Taking single label
                            label = np.copy(d[key])
                            label[label != val_label] = 1
                            label = 1 - label
                            # Label should be represented in 1
                            label = (label > 0.5).astype(np.float32)
                            # Taking single prediction
                            pred = np.copy(d[self.pred])
                            pred[pred != val_label] = 1
                            pred = 1 - pred
                            # Prediction should be represented in one
                            pred = (pred > 0.5).astype(np.float32)
                        all_discrepancies[key_label] = self._apply(label, pred)
                    d[self.discrepancy] = all_discrepancies
                    return d
                else:
                    print("This transform only applies to 'label' key")
            return d