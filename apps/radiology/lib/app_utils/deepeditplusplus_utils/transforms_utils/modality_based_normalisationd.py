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

Discard add guidance version information:

Version 0: The original deepedit implementation.
Version 1: The DeepEdit++ v1.1 implementation. (modality is taken from the dataset.json file rather than as an input argument to the main/run.py)  
Version 2: The DeepEdit++ v1.1.2 implementation (modality is taken from the dataset.json file rather than as an input argument to the main/run.py),
the non-CT normalisation is performed using hard clipped quartiles. 

Version 3: The DeepEdit++ v1.2 implementation ALSO takes the information about the normalisation of CT from the heuristic planner.
'''

class ImageNormalisationd(MapTransform):
    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False, 
                modality: str = "MRI",
                version_param: str = '1'
    ):
        '''
        Image normalisation transform which enables per-modality image normalisation.
        '''
        super().__init__(keys, allow_missing_keys)
        self.modality = modality
        self.version_param = version_param 

        self.supported_versions = ['0','1','2','3']
        self.x_ray_modalities = ['CT', 'X-Ray']

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)

        if self.version_param == '0':
            for key in self.key_iterator(d):
                
                #Default spleen ct only!
                d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
                
                d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]

            return d 

        if self.version_param == '1':

            for key in self.key_iterator(d):
                if self.modality == "CT":
                    #TODO: Consider changing this to ScaleIntensity or ScaleIntensityPercentile so that it just does it based off the percentiles in the image..
                    d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
                
                elif self.modality == "MRI":

                    d[key] = ScaleIntensity(minv=0.0, maxv=1.0)(d[key])#b_min=0.0, b_max=1.0, clip=True)(d[key])
                    # d[key] = NormalizeIntensity()(d[key])
                d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]

            return d  

            
        if self.version_param == '2':
            
            #Divided into X-ray/X-ray CT and non X-ray modalities.

            for key in self.key_iterator(d):

                if self.modality in self.x_ray_modalities:
                    #TODO: Consider changing this to ScaleIntensity or ScaleIntensityPercentile so that it just does it based off the percentiles in the image..
                    d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
                
                else:

                    # d[key] = ScaleIntensity(minv=0.0, maxv=1.0)(d[key])#b_min=0.0, b_max=1.0, clip=True)(d[key])
                    # d[key] = NormalizeIntensity()(d[key])
                    d[key] = ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True)(d[key])

                    # Values taken from :
                    # Does image normalization and intensity resolution impact texture classification? Marcin Kociołek , Michał Strzelecki, Rafał Obuchowicz.

                d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]

            return d  
        

        if self.version_param == '3':
            
            raise ValueError('This version is not yet supported! This is intended to have the CT modalities by normalised using the heuristic planner values')

            return d    