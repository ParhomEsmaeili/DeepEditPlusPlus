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
Add guidance signal transform is used for encoding the clicks provided into the class separated additional channels (alongside the input image).


Version 0: The original deepedit implementation.
Version 1: The DeepEdit++ v1.1 implementation. 
Version 2: The DeepEdit++ v1.1 implementation in TORCH. 

'''

class AddGuidanceSignalDeepEditd(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
        version_param: the version parameter which informs us which version of this transform we are using.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sigma: int = 3,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        label_names: dict | None = None,
        version_param: str = '1'
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.label_names = label_names or {}
        self.version_param = version_param 

        self.supported_versions = ['1']

        assert self.version_param in self.supported_versions, "The version being used is not supported for this transform!"

    def _get_signal(self, image, guidance):

        if self.version_param == "1":
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
        
        if self.version_param == "1":

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