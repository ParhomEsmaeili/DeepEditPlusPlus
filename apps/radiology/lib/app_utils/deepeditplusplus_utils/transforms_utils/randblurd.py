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
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import GaussianSmooth
import sys
import os
from os.path import dirname as up
sys.path.append(up(up(__file__)))
logger = logging.getLogger(__name__)

'''
version 1: Standard univariate gaussian implementation, with the kernel size being randomly generated.
'''

class UniformRandGaussianSmoothd(Randomizable, MapTransform):

    '''
    Similar dictionary wrapper as the RandGaussianSmoothd, HOWEVER, that implementation did not permit a single kernel parameter to be randomly generated. It would only gene
    -rate kernel sizes randomly in all spatial dimensions. This wrapper uses the same kernel size uniformly across each spatial dim. (albeit still randomly generated!)

    '''

    backend = GaussianSmooth.backend 

    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False, 
                kernel_bounds: tuple[float, float] = (0.5, 1),
                prob: float = 0.1,
                mask_key: Optional[str] = 'intensity_aug_mask',
                version_param : str = '1'
    ):

        super().__init__(keys, allow_missing_keys)
        self.kernel_size_bounds = kernel_bounds
        self.prob = prob
        self.version_param = version_param 

        self.supported_versions = ['1']

        if not self.version_param in self.supported_versions:
            raise ValueError("The version param is not supported/compatible.")
    
    def randomize(self, data=None):
        
        if self.version_param == '1':
            blur_bool = self.R.choice([True, False], p=[self.prob, 1.0 - self.prob])
            blur_sigma = self.R.uniform(self.kernel_size_bounds[0], self.kernel_size_bounds[1])
            
            return (blur_bool, blur_sigma)

    def __apply__(self, image, blur_bool, blur_sigma):
        
        if self.version_param == '1':
            if blur_bool:
                return GaussianSmooth(sigma=blur_sigma)(image)
            else:
                return image


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        
        
        if self.version_param == '1':
            
            (blur_bool, blur_sigma) = self.randomize() 

            for key in self.key_iterator(d):
        
                if key == "image":
                    
                    d[key] = self.__apply__(d[key], blur_bool, blur_sigma)
                
                else:
                    logger.info(f"Gaussian blur not intended for key: {key}")


            return d
        

if __name__ == '__main__':

    from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, LoadImaged, DivisiblePadd
    from transforms_utils.modality_based_normalisationd import ImageNormalisationd 
    import nibabel as nib 

    input_dict = {'image': '/home/parhomesmaeili/DeepEditPlusPlus Development/DeepEditPlusPlus/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTr/BraTS2021_00000.nii.gz'}

    load_stack_base = [LoadImaged(keys=("image"), reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=("image")), 
                Orientationd(keys=("image"), axcodes="RAS"), 
                ImageNormalisationd(keys="image", modality="MRI", version_param='4')]

    load_stack = load_stack_base + [DivisiblePadd(keys="image", k=[64,64,32], mode='reflection')]

    blur_stack = load_stack_base +  [UniformRandGaussianSmoothd(keys="image", prob=1), DivisiblePadd(keys="image", k=[64,64,32], mode='reflection')]
    

    output_load = Compose(load_stack)(input_dict)
    output_blur = Compose(blur_stack)(input_dict)

    print('fin!')