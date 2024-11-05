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

Simulate editing clicks in a probabilistic manner.

Version 0: The original deepedit implementation.
Version 1: The DeepEdit++ v1.1 implementation. 
Version 2: The DeepEdit++ v1.1 implementation in TORCH. 

'''


class AddRandomGuidanceDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
        version_param: The version of the transform that we are currently using.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        allow_missing_keys: bool = False,
        version_param : str = '1'
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
        self.version_param = version_param 

        self.supported_versions = ['1']
        assert self.version_param in self.supported_versions

    def randomize(self, data=None):
        probability = data[self.probability]
        self.will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        
        if self.version_param == '0' or self.version_param == '1':
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

        if self.version_param == '0' or self.version_param == '1':

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
        
        
        if self.version_param == '1':
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