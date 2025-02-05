# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

#########################################################
import nibabel as nib
import os 
from monai.utils import MetaKeys

class DiscardAddGuidanced(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        number_intensity_ch: int = 1,
        probability: float = 1.0,
        label_names: Sized | None = None,
        allow_missing_keys: bool = False,
    ):
        """
        Discard positive and negative points according to discard probability

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            number_intensity_ch: number of intensity channels
            probability: probability of discarding clicks
        """
        super().__init__(keys, allow_missing_keys)

        self.number_intensity_ch = number_intensity_ch
        self.discard_probability = probability
        self.label_names = label_names or []

    def _apply(self, image):
        if self.discard_probability >= 1.0 or np.random.choice(
            [True, False], p=[self.discard_probability, 1 - self.discard_probability]
        ):
            signal = np.zeros(
                (len(self.label_names), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32
            )
            if image.shape[0] == self.number_intensity_ch + 2 * len(self.label_names):
                image[self.number_intensity_ch: self.number_intensity_ch + len(self.label_names), ...] = signal
            else:
                image = np.concatenate([image, signal], axis=0)
        return image

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                print(d[key].shape)
                tmp_image = self._apply(d[key])
                if isinstance(d[key], MetaTensor):
                    print(tmp_image.shape)
                    d[key].array = tmp_image
                else:
                    d[key] = tmp_image
            else:
                print("This transform only applies to the image")
        return d


class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(
        self, keys: KeysCollection, label_names: dict[str, int] | None = None, allow_missing_keys: bool = False
    ):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names or {}

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
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


# class SingleLabelSelectiond(MapTransform):
#     def __init__(
#         self, keys: KeysCollection, label_names: Sequence[str] | None = None, allow_missing_keys: bool = False
#     ):
#         """
#         Selects one label at a time to train the DeepEdit

#         Args:
#             keys: The ``keys`` parameter will be used to get and set the actual data item to transform
#             label_names: all label names
#         """
#         super().__init__(keys, allow_missing_keys)

#         self.label_names: Sequence[str] = label_names or []
#         self.all_label_values = {
#             "spleen": 1,
#             "right kidney": 2,
#             "left kidney": 3,
#             "gallbladder": 4,
#             "esophagus": 5,
#             "liver": 6,
#             "stomach": 7,
#             "aorta": 8,
#             "inferior vena cava": 9,
#             "portal_vein": 10,
#             "splenic_vein": 11,
#             "pancreas": 12,
#             "right adrenal gland": 13,
#             "left adrenal gland": 14,
#         }

#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
#         d: dict = dict(data)
#         for key in self.key_iterator(d):
#             if key == "label":
#                 # Taking one label at a time
#                 t_label = np.random.choice(self.label_names)
#                 d["current_label"] = t_label
#                 d[key][d[key] != self.all_label_values[t_label]] = 0.0
#                 # Convert label to index values following label_names argument
#                 max_label_val = self.label_names.index(t_label) + 1
#                 d[key][d[key] > 0] = max_label_val
#                 print(f"Using label {t_label} with number: {d[key].max()}")
#             else:
#                 warnings.warn("This transform only applies to the label")
#         return d


class AddGuidanceSignalDeepEditd(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sigma: int = 3,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        label_names: dict | None = None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.label_names = label_names or {}

    def _get_signal(self, image, guidance):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance

        # In inference the user may not provide clicks for some channels/labels
        if len(guidance):
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            else:
                signal = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)

            sshape = signal.shape
            for point in guidance:  # TO DO: make the guidance a list only - it is currently a list of list
                if np.any(np.asarray(point) < 0):
                    continue

                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if np.max(signal[0]) > 0:
                signal_tensor = torch.tensor(signal[0])
                pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                signal_tensor = signal_tensor.squeeze(0).squeeze(0)
                signal[0] = signal_tensor.detach().cpu().numpy()
                signal[0] = (signal[0] - np.min(signal[0])) / (np.max(signal[0]) - np.min(signal[0]))
            return signal
        else:
            if dimensions == 3:
                signal = np.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
            else:
                signal = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)
            return signal

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        if "will_interact" in d.keys():
            #If there a "will_interact" entry in dict then use the one from the data dictionary. This is intended to be used for the inner loop of training.
            will_interact = d["will_interact"]
        else: 
            #If there is not, then use the default that it will interact. This is used for the pre-transforms and inference, since we always want guidance to be added
            will_interact = True 
            

        if will_interact:
            for key in self.key_iterator(d):
                if key == "image":
                    image = d[key]
                    tmp_image = image[0 : 0 + self.number_intensity_ch, ...]
                    #logger.info(f"Dimensions of Image Pre-Guidance are {tmp_image.shape}")
                    guidance = d[self.guidance]
                    
                    printing_guidance = dict()

                    for key_label in guidance.keys():
                        # Getting signal based on guidance
                        signal = self._get_signal(image, guidance[key_label])
                        #logger.info(f"Guidance signal dimensions are {signal.shape}")
                        tmp_image = np.concatenate([tmp_image, signal], axis=0)


                        # tmp_guidance = guidance[key_label].tolist() if isinstance(guidance[key_label], np.ndarray) else guidance[key_label]
                        # tmp_guidance = json.loads(guidance[key_label]) if isinstance(guidance[key_label], str) else guidance[key_label]
                        # printing_guidance[key_label] = [point for point in tmp_guidance if np.all(np.asarray(point) >= 0)]

                        if isinstance(d[key], MetaTensor):
                            d[key].array = tmp_image
                        else:
                            d[key] = tmp_image
                    #logger.info(f'Guidance points are {printing_guidance}')
                    return d
                else:
                    print("This transform only applies to image key")
        return d


class FindAllValidSlicesDeepEditd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.

    Args:
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids: Hashable = "sids", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            sids[key_label] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d


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
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid, key_label):
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


class FindDiscrepancyRegionsDeepEditd(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred: str = "pred",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred = pred
        self.discrepancy = discrepancy

    @staticmethod
    def disparity(label, pred):
        disparity = label - pred
        # Negative ONES mean predicted label is not part of the ground truth
        # Positive ONES mean predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
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


class AddRandomGuidanceDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self.will_interact = None
        self.is_pos: bool | None = None
        self.is_other: bool | None = None
        self.default_guidance = None
        self.guidance: dict[str, list[list[int]]] = {}

    def randomize(self, data=None):
        probability = data[self.probability]
        self.will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance.flatten()) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, guidance, discrepancy, label_names, labels):
        # Positive clicks of the segment in the iteration
        pos_discr = discrepancy[0]  # idx 0 is positive discrepancy and idx 1 is negative discrepancy

        # Check the areas that belong to other segments
        other_discrepancy_areas = {}
        for _, (key_label, val_label) in enumerate(label_names.items()):
            if key_label != "background":
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 0
                tmp_label = (tmp_label > 0.5).astype(np.float32)
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label)
            else:
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 1
                tmp_label = 1 - tmp_label
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label)

        # Add guidance to the current key label
        if np.sum(pos_discr) > 0:
            guidance.append(self.find_guidance(pos_discr))
            self.is_pos = True

        # Add guidance to the other areas
        for key_label in label_names.keys():
            # Areas that cover more than 50 voxels
            if other_discrepancy_areas[key_label] > 50:
                self.is_other = True
                if key_label != "background":
                    tmp_label = np.copy(labels)
                    tmp_label[tmp_label != label_names[key_label]] = 0
                    tmp_label = (tmp_label > 0.5).astype(np.float32)
                    self.guidance[key_label].append(self.find_guidance(discrepancy[1] * tmp_label))
                else:
                    tmp_label = np.copy(labels)
                    tmp_label[tmp_label != label_names[key_label]] = 1
                    tmp_label = 1 - tmp_label
                    self.guidance[key_label].append(self.find_guidance(discrepancy[1] * tmp_label))

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        discrepancy = d[self.discrepancy]
        d[self.guidance_key] = {} 
        self.randomize(data)

        d["will_interact"] = self.will_interact

        if self.will_interact:

            # Convert all guidance to lists so new guidance can be easily appended. These two loops are kept separate since we need an instantiated set of guidance lists for the current add_guidance function.
            for key_label in d["label_names"].keys():
                
                self.guidance[key_label] = [] #This means that we can completely reset the guidance list that was initially provided for generating intial segmentations.
                
                # else:
                #     tmp_gui = guidance[key_label]
                #     tmp_gui = tmp_gui.tolist() if isinstance(tmp_gui, np.ndarray) else tmp_gui
                #     tmp_gui = json.loads(tmp_gui) if isinstance(tmp_gui, str) else tmp_gui
                #     self.guidance[key_label] = [j for j in tmp_gui if -1 not in j] #This -1 logic deletes all the initial seeds that are not legitimate (i.e. the default set ones which had values of -1)

            # Add guidance according to discrepancy
            for key_label in d["label_names"].keys():
                
                # Add guidance based on discrepancy
                self.add_guidance(self.guidance[key_label], discrepancy[key_label], d["label_names"], d["label"])
            
            # Checking the number of clicks
            num_clicks = random.randint(1, 10)
            
            counter = 0
            keep_guidance = []
            while True:
                aux_label = random.choice(list(d["label_names"].keys()))
                if aux_label in keep_guidance:
                    pass
                else:
                    keep_guidance.append(aux_label)
                    counter = counter + len(self.guidance[aux_label])
                    # If collected clicks is bigger than max clicks, discard the others
                    if counter >= num_clicks:
                        for key_label in d["label_names"].keys():
                            if key_label not in keep_guidance:
                                self.guidance[key_label] = []
                        logger.info(f"Number of simulated clicks: {counter}")
                        #logger.info(f"Final Guidance points generated: {self.guidance}")
                        break

                # Breaking once all labels are covered
                if len(keep_guidance) == len(d["label_names"].keys()):
                    logger.info(f"Number of simulated clicks: {counter}")
                    #logger.info(f"Final Guidance points generated: {self.guidance}")
                    break
            d[self.guidance_key] = self.guidance  # Update the guidance
        return d


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

    """

    def __init__(
        self,
        ref_image: str,
        guidance: str = "guidance",
        label_names: dict | None = None,
        meta_keys: str | None = None,
        meta_key_postfix: str = "meta_dict",
    ):
        self.ref_image = ref_image
        self.guidance = guidance
        self.label_names = label_names or {}
        self.meta_keys = meta_keys
        self.meta_key_postfix = meta_key_postfix

    @staticmethod
    def _apply(clicks, factor):
        if len(clicks):
            guidance = np.multiply(clicks, factor).astype(int).tolist()
            return guidance
        else:
            return []

    def __call__(self, data):
        d = dict(data)
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


class ResizeGuidanceMultipleLabelDeepEditd(Transform):
    """
    Resize the guidance based on cropped vs resized image.

    """

    def __init__(self, guidance: str, ref_image: str) -> None:
        self.guidance = guidance
        self.ref_image = ref_image

    def __call__(self, data):
        d = dict(data)
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


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This is only for pred key")
        return d


class AddInitialSeedPointMissingLabelsd(Randomizable, MapTransform):
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
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid):
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

        label_guidance = []
        # If there are is presence of that label in this slice
        if np.max(blobs_labels) <= 0:
            label_guidance.append(self.default_guidance)
        else:
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).astype(np.float32)
                    if np.sum(label) == 0:
                        label_guidance.append(self.default_guidance)
                        continue

                # The distance transform provides a metric or measure of the separation of points in the image.
                # This function calculates the distance between each pixel that is set to off (0) and
                # the nearest nonzero pixel for binary images
                # http://matlab.izmiran.ru/help/toolbox/images/morph14.html
                distance = distance_transform_cdt(label).flatten()
                probability = np.exp(distance) - 1.0

                idx = np.where(label.flatten() > 0)[0]
                seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                dst = distance[seed]

                g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
                g[0] = dst[0]  # for debug
                if dimensions == 2 or dims == 3:
                    label_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    label_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

        return np.asarray(label_guidance)

    def _randomize(self, d, key_label):
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
                        self._apply(tmp_label, self.sid.get(key_label)).astype(int).tolist()
                    )
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
        return d


class FindAllValidSlicesMissingLabelsd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.
    Args:
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids: Hashable = "sids", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            # If there are not slices with the label
            if l_ids == []:
                l_ids = [-1] * 10
            sids[key_label] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d

##################################################################################################################################################
class AddSegmentationInputChannels(Randomizable, MapTransform):
    '''
    Generates the additional channels to concatenate with the image tensor after the guidance channels, representing the "previous segmentation" split by class.

    '''
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, previous_seg_name:str | None = None, number_intensity_ch : int = 1, label_names: dict | None = None, previous_seg_flag: bool = False):
        super().__init__(keys, allow_missing_keys)
        
        self.previous_seg_name = previous_seg_name
        self.number_intensity_ch = number_intensity_ch
        self.label_names = label_names or {}
        self.previous_seg_flag = previous_seg_flag
        #self.previous_seg = previous_seg 
    
    def randomize(self, image):
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        random_array = self.R.choice(list(self.label_names.values()), image.shape[1:])
        # print(np.unique(random_array, return_counts=True))
        return random_array
        
        #raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


    def _get_mask(self, image, previous_seg):
        '''
        get the mask according to the previous_Segmentation_flag, if in the Editing mode then it is the previous segmentation, if it is the other two modes (AutoSeg/Growing From Prompts) then it should
        be randomly generated..

        previous segmentation is assumed to be a single tensor or None, not k-channels separated by class, for Editing this function splits the input image. we assume that the tensor has discrete values which represent the integer codes for the classes (discrete) (CURRENT METHOD DOESNT USE CONFIG INTEGER CODES LIKE THE SAVED IMAGES.).
        
        currently, self.labels key:values are class:integer codes, but the previous segmentation image is imported with values of 0 - k-1, instead of with the correct integer codes(hence the currently implementation for the DeepEdit loop)

        '''
        output_signal = [] #initialise the list which we will concatenate to. 
        
        if self.previous_seg_flag:
            
            for key in self.label_names.keys():
                output_signal.append(np.where(previous_seg == self.label_names[key], 1, 0)) 
                
            return np.stack(output_signal, dtype=np.float32)    
                
        else:
            #TODO Generate a HWD array where each voxel is randomly sampling uniformly from classes 1:k with probabilty 1/k. Use logical arrays to split into k channel tensors.
            random_array = self.randomize(image)
            
            for key in self.label_names.keys():
                output_signal.append(np.where(random_array == self.label_names[key], 1, 0))
                
            return np.stack(output_signal, dtype=np.float32)
        

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)

        if "will_interact" in d.keys():
            #If there a "will_interact" entry in dict then use the one from the data dictionary. This is intended to be used for the inner loop of training.
            will_interact = d["will_interact"]
        else: 
            #If there is not, then use the default that it will interact. This is used for the pre-transforms and inference, since we always want guidance to be added
            will_interact = True

        if will_interact:
            for key in self.key_iterator(d):
                if key == "image":
                    image = np.copy(d[key])
                    
                    n_dims = len(image[0].shape)
                    
                    if self.previous_seg_flag: #If in inference for example , where the previous segmentation should still be a (n + 1)D tensor where n = spatial dimensions of image:  
                        if len(d[self.previous_seg_name].shape) == n_dims + 1:
                            previous_seg = d[self.previous_seg_name].squeeze()
                        elif len(d[self.previous_seg_name].shape) == n_dims: #If the previous segmentation is already an n * D tensor where n = spatial dimensions of image
                            previous_seg = d[self.previous_seg_name]
                        #TODO: Delete this temporary check.    
                        #nib.save(nib.Nifti1Image(np.array(previous_seg), None), os.path.join('/home/parhomesmaeili/TrainingInnerLoopPrediction/ActivatedPred.nii.gz'))
                    else:
                        previous_seg = None


                    #if label names is not inputted when instantiating the class, use the label names from the data dictionary.
                    if self.label_names:
                        pass
                    else:
                        self.label_names = d["label_names"]    
                    
                    tmp_image = image[0 : self.number_intensity_ch + len(self.label_names), ...]
            
                    # Getting signal
                    signal = self._get_mask(image, previous_seg) 

                    #logger.info(f"Dimensions of the split channels are {signal.shape}")
                    
                    tmp_image = np.concatenate([tmp_image, signal], axis=0, dtype=np.float32)
                    
                    # for i in range(tmp_image.shape[0]):
                    #     placeholder = tmp_image[i]
                    #     nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/AddingSegmentationChannels', str(i)+'.nii.gz'))

                    if isinstance(d[key], MetaTensor):
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                    return d
                else:
                    print("This transform only applies to image key")
        return d
    

class MappingLabelsInDatasetd(MapTransform):
    def __init__(
        self, keys: KeysCollection, original_label_names: dict[str, int] | None = None, label_names: dict[str, int] | None = None, label_mapping: dict[str, list] | None = None, allow_missing_keys: bool = False
    ):
        """
        Changing the labels from the original dataset, to what is in the config.csv or config text file. 

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)
        self.original_label_names = original_label_names
        self.label_names = label_names
        self.label_mapping = label_mapping 

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
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

# class ExtractChannelsd(MapTransform):
#     def __init__(
#         self, keys: KeysCollection, extract_channels:list, allow_missing_keys: bool = False
#     ):
#         """
#         Changing the labels from the original dataset, to what is in the config.csv or config text file. 

#         Args:
#             keys: The ``keys`` parameter will be used to get and set the actual data item to transform
#         """
#         super().__init__(keys, allow_missing_keys)
#         self.extract_channels = extract_channels

#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
#         d: dict = dict(data)
#         for key in self.key_iterator(d):
#             image = np.copy(d[key])
#             if len(self.extract_channels) == image.shape[0]:
#                 pass
#             else:
#                 delete_list = list(set([i for i in range(image.shape[0])]) ^ set(self.extract_channels))
#                 tmp_image = np.delete(image, delete_list, axis=0)
#                 if isinstance(d[key], MetaTensor):
#                     d[key].array = np.zeros(2,)
#                     #reset to a dummy array temporarily so that we can re-assign, immutability of numpy arrays means we cannot reduce an array in place unless we add new elements.
#                     d[key].array = tmp_image #np.stack([tmp_image[0]], axis=0) #np.stack([tmp_image[0], tmp_image[0], tmp_image[0], tmp_image[0], tmp_image[0]], axis=0)
#                 else:
#                     d[key] = np.zeros(2,)
#                     #dummy array TODO: verify that this actually works? or if it is even needed.
#                     d[key] = tmp_image

#         return d  

class ExtractMeta(MapTransform):
    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False
    ):
        """
        Extracting the meta information from the original state of the image.. 

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(d[key], MetaTensor):
                d["saved_meta"] = image.meta
        return d


class IntensityCorrection(MapTransform):
    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False, modality: str = "CT"
    ):
        '''
        Intensity rescaling function which provides the flexibility to rescale based off which modality the image is in the request to the MonaiLabelApp.
        '''
        super().__init__(keys, allow_missing_keys)
        self.modality = modality

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if self.modality == "CT":
                #TODO: Consider changing this to ScaleIntensity or ScaleIntensityPercentile so that it just does it based off the percentiles in the image..
                d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
            elif self.modality == "MRI":
                # d[key] = ClipIntensityPercentiles(1, 99)(d[key])
                d[key] = ScaleIntensity(minv=0.0, maxv=1.0)(d[key])#b_min=0.0, b_max=1.0, clip=True)(d[key])
                # d[key] = NormalizeIntensity()(d[key])
            d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]

        return d 
    
class MappingGuidancePointsd(MapTransform):
    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False, original_spatial_size_key:str = None, label_names: dict[str, int] = None, guidance:str="guidance"
    ):
        '''
        Function which maps the guidance points generated from original resolution RAS, to the padded images.
        '''
        super().__init__(keys, allow_missing_keys)
        self.original_spatial_size_key = original_spatial_size_key
        self.label_names = label_names 
        self.guidance = guidance
    
    def __apply__(self, guidance_point, current_size):
        pre_padding = [(current_size[i] - self.original_spatial_size[i]) // 2 for i in range(len(current_size))]
        return [j + pre_padding[i] for i,j in enumerate(guidance_point)] 

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            self.original_spatial_size = d[self.original_spatial_size_key]
            if key == "image":
                current_image_size = [j for i,j in enumerate(d[key].shape) if i!=0]
                #updated_guidance = dict() 

                for label_name in self.label_names.keys():
                    original_guidance_points = d[label_name]
                    updated_guidance_points = []
                    
                    for guidance_point in original_guidance_points:
                        updated_guidance_point = self.__apply__(guidance_point, current_size=current_image_size)
                        updated_guidance_points.append(updated_guidance_point)

                    d[label_name] = updated_guidance_points 

                all_guidances = {}
                for key_label in self.label_names.keys():
                    clicks = d.get(key_label, [])
                    #clicks = list(np.array(clicks).astype(int))
                    all_guidances[key_label] = clicks#.tolist()
                d[self.guidance] = all_guidances

                return d
        
