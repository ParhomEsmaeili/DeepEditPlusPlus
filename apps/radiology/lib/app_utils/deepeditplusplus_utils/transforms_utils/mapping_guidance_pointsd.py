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
Mapping guidance points from where they're placed in inference (in RAS assumed) to the formatting after pre-processing.

Version 1: The DeepEdit++ v1.1 implementation. for instances where the images are being padded during pre-processing (and the original points are placed on the non-padded images..).

'''

class MappingGuidancePointsd(MapTransform):
    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False, 
                original_spatial_size_key:str = None, 
                label_names: dict[str, int] = None, 
                guidance:str="guidance",
                version_param : str = '1'
    ):
        '''
        Function which maps the guidance points generated from original resolution RAS, to the padded images.
        '''
        super().__init__(keys, allow_missing_keys)
        self.original_spatial_size_key = original_spatial_size_key
        self.label_names = label_names 
        self.guidance = guidance
        self.version_param = version_param 

        self.supported_versions = ['1']

        assert self.version_param in self.supported_versions
    
    def __apply__(self, guidance_point, current_size):
        
        if self.version_param == '1':
            pre_padding = [(current_size[i] - self.original_spatial_size[i]) // 2 for i in range(len(current_size))]
            return [j + pre_padding[i] for i,j in enumerate(guidance_point)] 


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        
        
        if self.version_param == '1':

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