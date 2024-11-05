import copy
import logging
from typing import Dict, Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms import Randomizable
from monai.transforms.transform import MapTransform, Transform

logger = logging.getLogger(__name__)

class GetCentroidsd(MapTransform):
    def __init__(self, keys: KeysCollection, centroids_key: str = "centroids", version_param: str = '0', allow_missing_keys: bool = False):
        """
        Get centroids

        :param keys: The ``keys`` parameter will be used to get and set the actual data item to transform

        """
        super().__init__(keys, allow_missing_keys)
        self.centroids_key = centroids_key
        self.version_param = version_param 

        supported_versions = ['0']

        assert self.version_param in supported_versions

    def _get_centroids(self, label):
        
        if self.version_param == '0':
            centroids = []
            # loop over all segments
            areas = []
            for seg_class in np.unique(label):
                c = {}
                # skip background
                if seg_class == 0:
                    continue
                # get centre of mass (CoM)
                centre = []
                for indices in np.where(label == seg_class):
                    avg_indices = np.average(indices).astype(int)
                    centre.append(avg_indices)
                c[f"label_{int(seg_class)}"] = [int(seg_class), centre[-3], centre[-2], centre[-1]]
                centroids.append(c)
            return centroids

    def __call__(self, data):
        d: Dict = dict(data)
        
        if self.version_param == '0':
            for key in self.key_iterator(d):
                # Get centroids
                d[self.centroids_key] = self._get_centroids(d[key])
            return d