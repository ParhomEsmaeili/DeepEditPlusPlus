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
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
import sys
import os
from os.path import dirname as up
sys.path.append(up(up(__file__)))


logger = logging.getLogger(__name__)

'''
version 1: Standard approach as used by nnu-net for simulation of gamma adjustment. Wraps the AdjustContrast transform in monai. It is implemented in this manner
for ease of understanding. Alternatively, a stack of RandAdjustContrastd could be used in transforms list, but with the configurations set in the manner required for nnUnet 
(as implemented here)
version 2: Implements a masking so that the augmentation is only implemented on the foreground voxels.
'''

class RandGammaAdjustnnUNetd(Randomizable, MapTransform):
    """
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image_prob: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        foreground_info_key: The key used to extract information about whether the transform is applied to the entirety of the image or a subcomponent according to a mask.
        version_param: The version of the transform being used.
    """

    backend = AdjustContrast.backend

    def __init__(
        self,
        keys:KeysCollection,
        allow_missing_keys:bool = False,
        gamma_no_inv: Sequence[float] = (0.7, 1.5),
        gamma_with_inv: Sequence[float] = (0.7, 1.5),
        no_inv_gamma_prob: float = 0.1,
        with_inv_gamma_prob: float = 0.3,
        retain_stats: bool = True,
        foreground_info_key: Optional[str] = 'intensity_aug_mask_dict',
        version_param: str = '2',       
    ) -> None:
    #    RandomizableTransform .__init__(self, prob)
        super().__init__(keys, allow_missing_keys)

        supported_version_params = ['1', '2']
        
        self.version_param = version_param 

        if self.version_param not in supported_version_params:
            raise ValueError("This version of the transform is not supported")

        if len(gamma_no_inv) != 2:
            raise ValueError("gamma should be a number or pair of numbers.")
        else:
            self.gamma_no_inv = (min(gamma_no_inv), max(gamma_no_inv))

        if len(gamma_with_inv) != 2:
            raise ValueError("gamma should be a number or pair of numbers.")
        else:
            self.gamma_with_inv = (min(gamma_with_inv), max(gamma_with_inv))

        self.retain_stats: bool = retain_stats

        self.no_inv_gamma_prob = no_inv_gamma_prob
        self.with_inv_gamma_prob = with_inv_gamma_prob 
        self.foreground_info_key = foreground_info_key

        #We use dummy value of 0 here to initialise the class, so that we do not have to keep re-initialising the class during call.
        self.adjust_contrast_no_invert = AdjustContrast(
            gamma=0, invert_image=False, retain_stats=self.retain_stats
            )

        self.adjust_contrast_with_invert = AdjustContrast(
            gamma=0, invert_image=True, retain_stats=self.retain_stats
            )

    def randomize(self, data: Any | None = None) -> None:
        
        if self.version_param == '1' or self.version_param == '2':
            #Alternative method for doing the bool to check whether transform is done. 
            self.no_inv_bool = self.R.choice([True, False], p=[self.no_inv_gamma_prob, 1 - self.no_inv_gamma_prob])
            
            self.gamma_no_inv_val = self.R.uniform(low=self.gamma_no_inv[0], high=self.gamma_no_inv[1])
            
            self.with_inv_bool = self.R.choice([True, False], p=[self.with_inv_gamma_prob, 1 - self.with_inv_gamma_prob])

            self.gamma_with_inv_val = self.R.uniform(low=self.gamma_with_inv[0], high=self.gamma_with_inv[1])

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        
        d = dict(data) 

        if self.version_param == '1':
            
            self.randomize()
            
            if self.gamma_no_inv_val is None or self.gamma_with_inv_val is None:
                raise RuntimeError("gamma_value is not set, please call `randomize` function first.")

            for key in self.key_iterator(d):
        
                if key == "image":

                    if not self.no_inv_bool and not self.with_inv_bool:
                        return d
                    
                    if self.no_inv_bool and self.with_inv_bool:
                        d[key] = self.adjust_contrast_no_invert(img=d[key], gamma=self.gamma_no_inv_val)
                        d[key] = self.adjust_contrast_with_invert(img=d[key], gamma=self.gamma_with_inv_val)

                    elif self.no_inv_bool and not self.with_inv_bool:
                        d[key] = self.adjust_contrast_no_invert(img=d[key], gamma=self.gamma_no_inv_val)

                    elif not self.no_inv_bool and self.with_inv_bool:
                        d[key] = self.adjust_contrast_with_invert(img=d[key], gamma=self.gamma_with_inv_val)
                else:
                    logger.info(f'Not supported for key: {key}')
            return d

        if self.version_param == '2':
            self.randomize()
            if self.gamma_no_inv_val is None or self.gamma_with_inv_val is None:
                raise RuntimeError("gamma_value is not set, please call `randomize` function first.")

            for key in self.key_iterator(d):
        
                if key == "image":
                    
                    if not self.no_inv_bool and not self.with_inv_bool:
                        return d
                    
                    foreground_info_dict = d[self.foreground_info_key]

                    if foreground_info_dict == None:
                        raise ValueError('There should be a sub-dictionary containing the information regarding whether foreground only is being used.')

                    elif not foreground_info_dict['foreground_stats_only']:
                        #Just uses the existing approach which uses the entirety of the image for performing the transform.
                        if self.no_inv_bool and self.with_inv_bool:
                            d[key] = self.adjust_contrast_no_invert(img=d[key], gamma=self.gamma_no_inv_val)
                            d[key] = self.adjust_contrast_with_invert(img=d[key], gamma=self.gamma_with_inv_val)

                        elif self.no_inv_bool and not self.with_inv_bool:
                            d[key] = self.adjust_contrast_no_invert(img=d[key], gamma=self.gamma_no_inv_val)

                        elif not self.no_inv_bool and self.with_inv_bool:
                            d[key] = self.adjust_contrast_with_invert(img=d[key], gamma=self.gamma_with_inv_val)
                    
                    elif foreground_info_dict['foreground_stats_only']: 
                        #In this case, the values used for performing this transform are only taken from the mask region also (i.e. for the clamping the range of values
                        #for example)

                        #We extract the masked region 
                        raise NotImplementedError('This implementation is still not complete!')
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

    gamma_stack = load_stack_base +  [RandGammaAdjustnnUNetd(keys="image", no_inv_gamma_prob=0, with_inv_gamma_prob=1), DivisiblePadd(keys="image", k=[64,64,32])]
    

    output_load = Compose(load_stack)(input_dict)
    output_gamma = Compose(gamma_stack)(input_dict)

    print('fin!')