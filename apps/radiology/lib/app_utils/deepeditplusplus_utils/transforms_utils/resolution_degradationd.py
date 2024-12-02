# from __future__ import annotations

# import json
# import logging
# import random
# import warnings
# from collections.abc import Hashable, Mapping, Sequence, Sized
# from typing import Any, Optional
# import numpy as np
# import torch

# from monai.config import KeysCollection
# from monai.data import MetaTensor
# from monai.transforms.transform import MapTransform, Randomizable, Transform
# from monai.utils import min_version, optional_import
# from monai.transforms import Resize
# from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
# import sys
# import os
# from os.path import dirname as up


# sys.path.append(up(up(__file__)))

# logger = logging.getLogger(__name__)

# '''
# version 1: Standard approach as used by nnu-net for simulation of lower resolution images (improving the robustness in these examples).
# '''

# class ResolutionDegradationd(Randomizable, MapTransform):

#     backend = Resize.backend 

#     def __init__(self, 
#                 keys: KeysCollection, 
#                 allow_missing_keys: bool = False, 
#                 down_strategy: dict[str, str | bool] = {'mode':'nearest-exact', 'align': False},
#                 up_strategy: dict[str, str | bool] = {'mode':'bicubic', 'align': True},
#                 resampling_factor_bound: Sequence[int] = (1,2),
#                 degrade_prob: float = 0.125, 
#                 version_param : str = '1'
#     ):

#         super().__init__(keys, allow_missing_keys)
#         self.downsampling_strategy = down_strategy 
#         self.upsampling_strategy = up_strategy
#         self.resampling_factor_bound = resampling_factor_bound  
#         self.degrade_prob = degrade_prob
#         self.version_param = version_param 
        
#         self.supported_versions = ['1']

#         self.align_corners_versions = ['linear', 'bilinear', 'bicubic', 'trilinear']

#         if self.version_param not in self.supported_versions:
#             raise ValueError("Not a supported version")

#         if self.downsampling_strategy['mode'] in self.align_corners_versions:
#             if self.downsampling_strategy['align'] == None:
#                 raise ValueError("Align corners parameter required.")
        
#         if self.upsampling_strategy['mode'] in self.align_corners_versions: 
#             if self.upsampling_strategy['align'] == None:
#                 raise ValueError("Align corners parameter required.")
        
#     def randomize(self, data=None):
        
#         if self.version_param == '1':
#             resample_bool = self.R.choice([True, False], p=[self.degrade_prob, 1.0 - self.degrade_prob])
#             resample_factor = self.R.randint(self.resampling_factor_bound[0], self.resampling_factor_bound[1] + 1) #Exclusive of the upper bound.
            
#             return (resample_bool, resample_factor)

#     def __apply__(self, image, resample_bool, resample_factor):
        
#         if self.version_param == '1':

#             resample_size = [i//resample_factor for i in image.shape[1:]]
#             original_size = list(image.shape[1:])

#             if resample_bool:
                
#                 if self.downsampling_strategy['mode'] in  self.align_corners_versions:     
#                     downsize = Resize(spatial_size=resample_size, mode=self.downsampling_strategy['mode'], align_corners=self.downsampling_strategy['align'])(image)
#                 else:
#                     downsize = Resize(spatial_size=resample_size, mode=self.downsampling_strategy['mode'])(image)
                
#                 if self.upsampling_strategy['mode'] in self.align_corners_versions:
#                     upsize = Resize(spatial_size=original_size, mode=self.upsampling_strategy['mode'], align_corners=self.upsampling_strategy['align'])(downsize)
#                 else:
#                     upsize = Resize(spatial_size=original_size, mode=self.upsampling_strategy['mode'])(downsize)            
#                 return upsize

#             else:
#                 return image


#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
#         d: dict = dict(data)
        
        
#         if self.version_param == '1':
            
#             (resample_bool, resample_factor) = self.randomize() 

#             for key in self.key_iterator(d):
        
#                 if key == "image":
                    
#                     d[key] = self.__apply__(d[key], resample_bool, resample_factor)
                
#                 else:
#                     logger.info(f"Resolution degradation not intended for key: {key}")


#             return d

# if __name__ == '__main__':

#     from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, LoadImaged
#     from transforms_utils.modality_based_normalisationd import ImageNormalisationd 

#     input_dict = {'image': '/home/parhomesmaeili/DeepEditPlusPlus Development/DeepEditPlusPlus/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTr/BraTS2021_00000.nii.gz'}

#     load_stack = [LoadImaged(keys=("image"), reader="ITKReader", image_only=False), 
#                 EnsureChannelFirstd(keys=("image")), 
#                 Orientationd(keys=("image"), axcodes="RAS"), 
#                 ImageNormalisationd(keys="image", modality="MRI", version_param='4'),
#                 ResolutionDegradationd(keys="image", degrade_prob=1)]
    
#     output = Compose(load_stack)(input_dict)

