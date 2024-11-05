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

Find the initial seeds from a ground truth

Version 0: The original deepedit implementation.


'''


class AddInitialSeedPointDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.

    Note that the label is of size (C, D, H, W) or (C, H, W)

    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)

    Args:
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
        version_param: the version of the transform that is being used.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
        version_param: str = "0"
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions
        self.version_param = version_param 

        self.supported_versions = ["0"]

        assert self.version_param in self.supported_versions 

    def _apply(self, label, sid, key_label):

        if self.version_param == "0": 

            dimensions = 3 if len(label.shape) > 3 else 2
            self.default_guidance = [-1] * (dimensions + 1)

            dims = dimensions
            if sid is not None and dimensions == 3:
                dims = 2
                label = label[0][..., sid][np.newaxis]  # Assume channel is first and depth is last CHWD

            # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
            label = (label > 0.5).astype(np.float32)
            # measure.label: Label connected regions of an integer array - Two pixels are connected
            # when they are neighbors and have the same value
            blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label
            if np.max(blobs_labels) <= 0:
                raise AssertionError(f"SLICES NOT FOUND FOR LABEL: {key_label}")

            pos_guidance = []
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).astype(np.float32)
                    if np.sum(label) == 0:
                        pos_guidance.append(self.default_guidance)
                        continue

                # The distance transform provides a metric or measure of the separation of points in the image.
                # This function calculates the distance between each pixel that is set to off (0) and
                # the nearest nonzero pixel for binary images - http://matlab.izmiran.ru/help/toolbox/images/morph14.html
                distance = distance_transform_cdt(label).flatten()
                probability = np.exp(distance) - 1.0

                idx = np.where(label.flatten() > 0)[0]
                seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                dst = distance[seed]

                g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
                g[0] = dst[0]  # for debug
                if dimensions == 2 or dims == 3:
                    pos_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    pos_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

            return np.asarray([pos_guidance])

    def _randomize(self, d, key_label):

        if self.version_param == "0":

            sids = d.get(self.sids_key).get(key_label) if d.get(self.sids_key) is not None else None
            sid = d.get(self.sid_key).get(key_label) if d.get(self.sid_key) is not None else None
            if sids is not None and sids:
                if sid is None or sid not in sids:
                    sid = self.R.choice(sids, replace=False)
            else:
                logger.info(f"Not slice IDs for label: {key_label}")
                sid = None
            self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        if self.version_param == "0":

            for key in self.key_iterator(d):
                if key == "label":
                    label_guidances = {}
                    for key_label in d["sids"].keys():
                        # Randomize: Select a random slice
                        self._randomize(d, key_label)
                        # Generate guidance base on selected slice
                        tmp_label = np.copy(d[key])
                        # Taking one label to create the guidance
                        if key_label != "background":
                            tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                        else:
                            tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                            tmp_label = 1 - tmp_label
                        label_guidances[key_label] = json.dumps(
                            self._apply(tmp_label, self.sid.get(key_label), key_label).astype(int).tolist()
                        )
                    d[self.guidance] = label_guidances
                    return d
                else:
                    print("This transform only applies to label key")
            return d

        