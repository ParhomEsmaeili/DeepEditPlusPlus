from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Hashable, Mapping, Sequence, Sized
from typing import Any, Optional
import numpy as np
import torch
from monai.config import KeysCollection, DtypeLike
from monai.data import MetaTensor
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import AdjustContrast
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

import sys
import os
from os.path import dirname as up
sys.path.append(up(up(__file__)))


logger = logging.getLogger(__name__)

'''
version 1: Standard approach as used by nnu-net for simulation of Gaussian noise. It samples from a uniform distribution for the selection of the variance,
instead of uniformly sampling from the standard deviation (non-linear transform -> means uniform sampling of variance != uniform sampling of std)
'''

class RandGaussianNoisennUNetd(Randomizable, MapTransform):
    """
    Add Gaussian noise to image.

    Args:
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        var: Variance bound which will be used when sampling standard deviation which parametrises the gaussian noise (if len = 1, then its just this value.)
        dtype: output data type, if None, same as input image. defaults to float32.
        sample_var: If True, sample the spread of the Gaussian distribution uniformly from 0 to var for the variance selected.

    Returns: Image with the same datatype as input, with gaussian noise potentially being added.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys:KeysCollection,
        allow_missing_keys:bool = False,
        prob: float = 0.1,
        mean: float = 0.0,
        var_bound: Sequence[float] = (0, 0.1),
        sample_var: bool = True,
        version_param: str = '1'
    ) -> None:
        
        super().__init__(keys, allow_missing_keys)

        self.prob = prob
        self.mean = mean
        self.var_bound = var_bound
        self.sample_var = sample_var
        self.version_param = version_param

        supported_version_params = ['1']
        if self.version_param not in supported_version_params:
            raise ValueError('This version parameter is not supported') 
        

    def randomize(self, img: NdarrayOrTensor) -> None:
        
        self.add_gaussian_bool = self.R.choice([True, False], p = [self.prob, 1 - self.prob])
        
        var = self.R.uniform(self.var_bound[0], self.var_bound[1]) if self.sample_var else self.var_bound[0]
        noise = self.R.normal(self.mean, var ** 0.5 , size=img.shape)
        # noise is float64 array, convert to the image dtype to save memory
        self.noise, *_ = convert_data_type(noise, dtype=img.dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        
        # img = convert_to_tensor(img, track_meta=get_track_meta())
        
        d = dict(data) 

        if self.version_param == '1':
            
            # img, *_ = convert_data_type(img, dtype=self.dtype)
            # noise, *_ = convert_to_dst_type(self.noise, img)

            for key in self.key_iterator(d):
                
                if key == "image":

                    self.randomize(d[key]) 

                    if self.noise is None:
                        raise RuntimeError("please call the `randomize()` function first.")

                    if self.add_gaussian_bool:
                        d[key] += self.noise

                    else: #If no noise being added.
                        return d 
                else:
                    logger.info(f'Not supported for key: {key}')
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

    load_stack = load_stack_base + [DivisiblePadd(keys="image", k=[64,64,32])]

    noise_stack = load_stack_base +  [RandGaussianNoisennUNetd(keys="image", prob=1), DivisiblePadd(keys="image", k=[64,64,32])]
    

    output_load = Compose(load_stack)(input_dict)
    output_noise = Compose(noise_stack)(input_dict)

    print('fin!')