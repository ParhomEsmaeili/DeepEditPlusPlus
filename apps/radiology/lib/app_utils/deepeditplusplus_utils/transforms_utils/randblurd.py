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
from monai.transforms import GaussianSmooth, SpatialPad, SpatialCrop
from scipy.ndimage import gaussian_filter
import sys
import os
from os.path import dirname as up
sys.path.append(up(up(__file__)))
logger = logging.getLogger(__name__)

'''
version 1: Standard univariate gaussian implementation, with the kernel size being randomly generated. Gaussian blur implementation is taken from monai which uses
the monai.nets gaussianfilter 

Version 2: nnU-net like standard univariate gaussian implementation with the kernel size being randomly generated. Gaussian blur implementation is mimicked from scipy 
so that we don't get the image border/band'ing issue. 

Version 3: nnU-net like standard univariate gaussian implementation with the kernel size being randomly generated. Gaussian blur implementation is taken directly from 
from scipy so that we don't get the image border/band'ing issue. 

'''

class UniformRandGaussianSmoothd(Randomizable, MapTransform):

    '''
    Similar dictionary wrapper as the RandGaussianSmoothd, HOWEVER, that implementation did not permit a single kernel parameter to be randomly generated. It would only gene
    -rate kernel sizes randomly in all spatial dimensions. This wrapper uses the same kernel size uniformly across each spatial dim. (albeit still randomly generated!)

    args:
        prob: The probability of performing this transform.
        mask_key: Optional key for the input data dictionary which may contain information about masking the transform (unlikely to be used)
        padding_mode: Optional mode information about the padding strategy implemented prior to blurring, in order to remove the banding/discontinuity effect that may occur with
        no padding.
        version_param: self explanatory.
    '''

    # backend = [TransformBackends.TORCH, TransformBackEnds.NUMPY] 

    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False, 
                kernel_bounds: tuple[float, float] = (0.5, 1),
                prob: float = 0.1,
                mask_key: Optional[str] = 'intensity_aug_mask',
                padding_mode: Optional[str] = 'reflect',
                version_param : str = '1'
    ):

        super().__init__(keys, allow_missing_keys)
        self.kernel_size_bounds = kernel_bounds
        self.prob = prob
        self.version_param = version_param 
        
        self.mask_key = mask_key
        self.padding_mode = padding_mode

        self.supported_versions = ['1', '2', '3']
        self.supported_padding_modes = ['reflect']

        if not self.version_param in self.supported_versions:
            raise ValueError("The version param is not supported/compatible.")

        if not self.padding_mode in self.supported_padding_modes:
            raise ValueError("The padding mode is not supported for this implementation")
        
    def randomize(self, data=None):
        
        if self.version_param in ['1', '2', '3']:
            blur_bool = self.R.choice([True, False], p=[self.prob, 1.0 - self.prob])
            blur_sigma = self.R.uniform(self.kernel_size_bounds[0], self.kernel_size_bounds[1])
            
            return (blur_bool, blur_sigma)

    def __apply__(self, image, blur_bool, blur_sigma):
        
        if self.version_param == '1':
            if blur_bool:
                return GaussianSmooth(sigma=blur_sigma)(image)
            else:
                return image

        if self.version_param == '2':
            
            # NotImplementedError('The implementation of the monai-version of the gaussian smoothing is not done. Need to consider how to select the cropping coordinates')
            if blur_bool:
                #We pad, blur, then crop.
                
                #Extracting the padding size as kernel size // 2:

                #We generate the dummy kernel, so that we know the kernel size...
                # kernel gaussian_1d(s, truncated=self.truncated, approx=self.approx)

                # spatialpad_class = SpatialPad()
                # spatialcrop_class = SpatialCrop()
                pass 

            else:
                return image

        if self.version_param == '3':

            if blur_bool:
                
                gaussian_filter(image, sigma=blur_sigma, mode=self.padding_mode, output=image)

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
        
        if self.version_param == '2':
            
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

    load_stack = load_stack_base + [DivisiblePadd(keys="image", k=[64,64,32], mode='reflect')]

    blur_stack = load_stack_base +  [UniformRandGaussianSmoothd(keys="image", prob=1), DivisiblePadd(keys="image", k=[64,64,32], mode='reflect')]
    

    output_load = Compose(load_stack)(input_dict)
    output_blur = Compose(blur_stack)(input_dict)

    print('fin!')