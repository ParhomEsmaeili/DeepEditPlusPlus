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

class AddRandomTestGuidanceCIMDeepEditd(Randomizable, MapTransform):
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
        version_param: str = '1',
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance
        self.discrepancy = discrepancy
        self.will_interact = None
        self.is_pos: bool | None = None
        self.is_other: bool | None = None
        self.default_guidance = None
        self.guidance: dict[str, list[list[int]]] = {}

    def randomize(self, data=None):
        # probability = data[self.probability]
        self.will_interact = True

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
        guidance = d[self.guidance_key]
        discrepancy = d[self.discrepancy]
        
        self.randomize(data)

        #TODO:



        raise ValueError("This script needs to be modified so that at least one new/unique click will be provided!")

        #TODO: Needs to be modified so that if any clicks are dropped out (due to them being placed before) that it has to restart the simulation until NO clicks are dropped out.
        #This is needed to ensure fairness across both cim and sim simulation strategies in the number of clicks being placed...

        if self.will_interact:
            # Convert all guidance to lists so new guidance can be easily appended
            for key_label in d["label_names"].keys():
                # tmp_gui = guidance[key_label]
                # tmp_gui = tmp_gui.tolist() if isinstance(tmp_gui, np.ndarray) else tmp_gui
                # tmp_gui = json.loads(tmp_gui) if isinstance(tmp_gui, str) else tmp_gui
                # self.guidance[key_label] = [j for j in tmp_gui if -1 not in j]
                self.guidance[key_label] = []
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
                        # logger.info(f"Number of simulated clicks: {counter}")
                        break

                # Breaking once all labels are covered
                if len(keep_guidance) == len(d["label_names"].keys()):
                    # logger.info(f"Number of simulated clicks: {counter}")
                    break
        #Appending the points generated that did not already exist in the existing set of points.

        for key_label in d["label_names"].keys():
            for guidance_point in self.guidance[key_label]:
                if guidance_point[1:] not in guidance[key_label]:
                    guidance[key_label].append(guidance_point)

        d[self.guidance_key] = guidance #self.guidance  # Update the guidance
        logger.info(f"Final number of simulation clicks: {sum([len(sublist) for sublist in list(guidance.values())])}")
        return d