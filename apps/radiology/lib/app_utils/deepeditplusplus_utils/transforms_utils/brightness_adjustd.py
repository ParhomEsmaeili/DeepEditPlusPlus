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
version 1: Standard approach as used by nnu-net for simulation of brightness adjustment.
'''

class RandBrightnessd(Randomizable, MapTransform):
    """
    Args:
        prob: Probability of brightness adjustment.
        bounds: Lower and Upper bound on the multiplier for brightness adjustment. 
        Mimicked behaviour taken from here: <https://github.com/MIC-DKFZ/batchgeneratorsv2/blob/master/batchgeneratorsv2/transforms/intensity/brightness.py>
    """

    backend = AdjustContrast.backend

    def __init__(
        self,
        keys:KeysCollection,
        allow_missing_keys:bool = False,
        prob: float = 0.15,
        bounds: Sequence[float] = (0.75, 1.25),
        version_param: str = '1',       
    ) -> None:
    #    RandomizableTransform .__init__(self, prob)
        super().__init__(keys, allow_missing_keys)

        supported_version_params = ['1']

        self.version_param = version_param 
        self.prob = prob
        if self.version_param not in supported_version_params:
            raise ValueError("This version of the transform is not supported")

        if len(bounds) != 2:
            raise ValueError("Multiplier bounds should be a number or pair of numbers.")
        else:
            self.bounds = (min(bounds), max(bounds))

    def randomize(self, data: Any | None = None) -> None:
        
        if self.version_param == '1':
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
                    d[key] = d[key] * self.multiplier
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

    brightness_stack = load_stack_base +  [RandBrightnessd(keys="image", prob=1), DivisiblePadd(keys="image", k=[64,64,32])]
    

    output_load = Compose(load_stack)(input_dict)
    output_brightness = Compose(brightness_stack)(input_dict)

    print('fin!')
