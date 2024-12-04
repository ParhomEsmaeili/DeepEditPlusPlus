from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Hashable, Mapping, Sequence, Sized
from typing import Any, Optional
import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import AdjustContrast
import sys
import os
from os.path import dirname as up
sys.path.append(up(up(__file__)))

logger = logging.getLogger(__name__)

'''
version 1: Standard approach as used by nnu-net for simulation of contrast adjustment, applies it to the entirety of the image.
version 2: Approach which is standardised by the values in the foreground prior to contrast adjustment, so that the foreground values remain close to their 
original distribution... Unlike the gamma transform equivalent, this should not be that catastrophic because the mean is just a flat bias that is being applied anyways.
Unlike the non-linear transform performed on a [0,1] normalised input (gamma contrast). BUT, it will not really clamp the voxel values within the foreground as much.

'''

class RandContrastAdjustd(Randomizable, MapTransform):
    """

    Image is assumed to be presented in CHWD format where C=1. 

    Args:
        prob: Probability of contrast adjustment.
        bounds: Lower and Upper bound on the multiplier for contrast adjustment. 
        preserve_range: Bool which controls whether the range is fixed (i.e. clamping the values that go beyond)
        version_param: Version of the class being used.
        This behaviour is mimicked
        from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
        <https://github.com/MIC-DKFZ/batchgeneratorsv2/blob/master/batchgeneratorsv2/transforms/intensity/contrast.py>`_
            
    """

    backend = AdjustContrast.backend

    def __init__(
        self,
        keys:KeysCollection,
        allow_missing_keys:bool = False,
        prob: float = 0.15,
        bounds: Sequence[float] = (0.75, 1.25),
        preserve_range: bool = True,
        foreground_info_key: Optional[str] = 'intensity_aug_mask_dict',
        version_param: str = '1',       
    ) -> None:
    #    RandomizableTransform .__init__(self, prob)
        super().__init__(keys, allow_missing_keys)

        supported_version_params = ['1', '2']
        
        self.prob = prob
        self.version_param = version_param
        if version_param not in supported_version_params:
            raise ValueError("This version of the transform is not supported")

        
        if len(bounds) != 2:
            raise ValueError("Multiplier bounds should be a number or pair of numbers.")
        else:
            self.bounds = (min(bounds), max(bounds))

        
        self.preserve_range = preserve_range
        self.foreground_info_key = foreground_info_key 

    def randomize(self, data: Any | None = None) -> None:
        
        if self.version_param == '1' or self.version_param == '2':
            self.adjust_bool = self.R.choice([True, False], p=[self.prob, 1 - self.prob])
            self.multiplier = self.R.uniform(low=self.bounds[0], high=self.bounds[1])
        

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        
        d = dict(data) 

        if self.version_param == '1':
            
            self.randomize()
            
            if not self.adjust_bool:
                return d
            
            for key in self.key_iterator(d):
                if key == "image":
                    
                    mean = d[key].mean()

                    if self.preserve_range:
                        minm = d[key].min()
                        maxm = d[key].max()

                    d[key] -= mean 
                    d[key] *= self.multiplier
                    d[key] += mean 

                    if self.preserve_range:
                        if type(d[key]) == torch.Tensor or type(d[key]) == MetaTensor:
                            d[key].clamp_(min=minm, max=maxm)
                        elif type(d[key]) == np.ndarray:
                            np.clip(d[key], a_min=minm, a_max=maxm, out=d[key])
                else:
                    logger.info(f'Not supported for key: {key}')
            return d

        elif self.version_param == '2':
            
            self.randomize()
            
            if not self.adjust_bool:
                return d
            
            for key in self.key_iterator(d):
                if key == "image":
                    
                    foreground_info_dict = d[self.foreground_info_key]

                    if foreground_info_dict['foreground_stats_only'] == None:
                        raise ValueError('There should be a sub-dictionary containing the information regarding whether foreground only is being used.')

                    elif not foreground_info_dict['foreground_stats_only']:

                        mean = d[key].mean()

                        if self.preserve_range:
                            minm = d[key].min()
                            maxm = d[key].max()

                        d[key] -= mean 
                        d[key] *= self.multiplier
                        d[key] += mean 

                        if self.preserve_range:
                            if type(d[key]) == torch.Tensor:
                                d[key].clamp(min=minm, max=maxm)
                            elif type(d[key]) == np.ndarray:
                                d[key].clip(min=minm, max=maxm)

                    elif foreground_info_dict['foreground_stats_only']:

                        #In this case, we only use the foreground voxels for extracting the mean/min/max used so that the foreground voxels are augmented more strongly.
                        #Otherwise, the mean will be dragged down by the background voxels, therefore eliminating the effect of the contrasting.

                        img = d[key][0,...] 
                        if img.shape != foreground_info_dict['foreground_region'].shape:
                            raise ValueError('The shape of the image and the foreground region mask were not identical')

                        if type(foreground_info_dict['foreground_region']) != torch.Tensor:
                            raise TypeError('The foreground region mask must be a torch tensor')
                            
                        if foreground_info_dict['foreground_stats_only']:
                            if foreground_mask.sum() != torch.nonzero(foreground_mask).shape[0]:
                                raise ValueError('The foreground mask did not sum to the quantity of foreground voxels')
                        else:
                            if int(foreground_mask.sum()) != torch.numel(foreground_mask):
                                raise ValueError('The foreground mask should be equal to the number of image voxels if foreground stats only = False')
                        
                        foreground_voxel_vals = torch.masked_select(img, foreground_info_dict['foreground_region'].bool())

                        mean = foreground_voxel_vals.mean()

                        #We only use the foreground voxel values for the mean. If we had used it for the min/max then it would clamp the background voxels in an 
                        #extremely atypical manner (i.e. the background value would not be background anymore..., it would be the minimum across foreground, which isn't
                        #how it should work at all..)

                        if self.preserve_range:
                            minm = d[key].min()
                            maxm = d[key].max()

                        d[key] -= mean 
                        d[key] *= self.multiplier
                        d[key] += mean 

                        if self.preserve_range:
                            if type(d[key]) == torch.Tensor or type(d[key]) == MetaTensor:
                                d[key].clamp_(min=minm, max=maxm)
                            elif type(d[key]) == np.ndarray:
                                np.clip(d[key], a_min=minm, a_max=maxm, out=d[key])

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
                ImageNormalisationd(keys="image", modality="MRI", version_param='5')]

    load_stack = load_stack_base + [DivisiblePadd(keys="image", k=[64,64,32], mode='reflect')]

    contrast_stack = load_stack_base +  [RandContrastAdjustd(keys="image", prob=1, version_param='2'), DivisiblePadd(keys="image", k=[64,64,32], mode='reflect')]
    

    output_load = Compose(load_stack)(input_dict)
    output_contrastadj = Compose(contrast_stack)(input_dict)

    print('fin!')