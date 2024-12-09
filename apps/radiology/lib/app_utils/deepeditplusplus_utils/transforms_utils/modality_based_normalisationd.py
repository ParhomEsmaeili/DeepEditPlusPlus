from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Hashable, Mapping, Sequence, Sized
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import ScaleIntensityRange, ScaleIntensityRangePercentiles, ScaleIntensity, NormalizeIntensity, ClipIntensityPercentiles, ForegroundMask
from scipy.ndimage import binary_fill_holes 
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
# measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)

# distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


'''

Version 0: The original deepedit implementation.
Version 1: The DeepEdit++ v1.1 implementation. (modality is taken from the dataset.json file rather than as an input argument to the main/run.py)  
Version 2: The DeepEdit++ v1.1.2 implementation (modality is taken from the dataset.json file rather than as an input argument to the main/run.py),
the non-CT normalisation is performed using hard clipped quartiles. 

Version 3: Non-CT normalisation is performed using Z-score based normalisation, ALSO takes the information about the normalisation of CT from the heuristic planner.
This Z-score based normalisation is implemented using the entirety of the image. 

Version 4: Modified version of version 3, but where the normalisation values are only extracted from the foreground regions, so that it doesn't get skewed by background.
However, it is implemented to the entirety of the image! 

Version 5: Modified version of version 4, but passes the foreground mask through such that it is saved for use in masking the intensity augmentations.
'''

class ImageNormalisationd(MapTransform):
    def __init__(self, 
                keys: KeysCollection, 
                allow_missing_keys: bool = False,
                planner_dict: Optional[dict] = None,
                invert: Optional[bool] = True, #This parameter assumes that the background/region we should ignore has smaller voxel intensity values. 
                modality: str = "MRI",
                foreground_info_key: Optional[str] = 'intensity_aug_mask_dict',
                version_param: str = '1'
    ):
        '''
        Image normalisation transform for X-ray/CT and non X-ray/CT modalities.

        args:
        planner_dict: An dictionary containing the parameters from the heuristic planner, required for implementations which use a X-ray/CT modality. 
        invert: An optional bool which by default is set to True. It determines whether the values in the image are brighter/larger for foreground regions (therefore 
        requiring inversion for the foreground mask extraction implementations).
        modality: The modality of the dataset.
        foreground_info_str : The key used to store the information about the foreground in the data dictionary.
        version_param: The version of the class that is being used. 

        Assumption is that the shape of the image is CHWD
        '''
        super().__init__(keys, allow_missing_keys)
        self.modality = modality
        self.version_param = version_param 
        self.invert = invert
        self.foreground_info_key = foreground_info_key
        self.supported_versions = ['0','1','2','3', '4', '5']
        self.x_ray_modalities = ['CT', 'X-Ray']
        
        if version_param not in self.supported_versions:
            raise ValueError("This version is not currently supported!")

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)

        if self.version_param == '0':
            for key in self.key_iterator(d):
                if key=='image':
                    #Default spleen ct only!
                    d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
                    
                    d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d 

        if self.version_param == '1':

            for key in self.key_iterator(d):

                if key=="image":
                    if self.modality == "CT":
                        #TODO: Consider changing this to ScaleIntensity or ScaleIntensityPercentile so that it just does it based off the percentiles in the image..
                        d[key] = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)(d[key])
                    
                    elif self.modality == "MRI":

                        d[key] = ScaleIntensity(minv=0.0, maxv=1.0)(d[key])#b_min=0.0, b_max=1.0, clip=True)(d[key])
                        # d[key] = NormalizeIntensity()(d[key])
                    d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d  

            
        if self.version_param == '2':
            
            #Divided into X-ray/X-ray CT and non X-ray modalities.

            for key in self.key_iterator(d):
                if key == 'image':
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
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d  
        
        
        if self.version_param == '3':
            
            #Divided into X-ray/X-ray CT and non X-ray modalities.

            for key in self.key_iterator(d):
                if key == 'image':
                    if self.modality in self.x_ray_modalities:
                        if planner_dict == None:
                            raise ValueError("The planner dictionary should not be a nonetype")
                        d[key] = torch.clamp(d[key], min = planner_dict['percentile_00_5_pix'], max = planner_dict['percentile_99_5_pix'])
                        d[key] = NormalizeIntensity(subtrahend=planner_dict['mean_pix'], divisor=planner_dict['std_pix'])(d[key])
                        #We can operate here on batch wise tensors because the normalisation is fixed across the entirety of the dataset.
                    else:
                        #We can also operate here on batch wise tensors if it is computing the divisors and subtrahend itself.
                        d[key] = NormalizeIntensity()(d[key])    
                    
                    d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d 

        if self.version_param == '4':
            
            #Divided into X-ray/X-ray CT and non X-ray modalities.

            for key in self.key_iterator(d):
                if key == 'image':
                    if self.modality in self.x_ray_modalities:
                        if planner_dict == None:
                            raise ValueError("The planner dictionary should not be a nonetype")
                        d[key] = d[key].clamp(min = planner_dict['percentile_00_5_pix'], max = planner_dict['percentile_99_5_pix'])

                        d[key] = NormalizeIntensity(subtrahend=planner_dict['mean_pix'], divisor=planner_dict['std_pix'])(d[key])
                        #We can operate here with a planner dict val because the normalisation is fixed across the entirety of the dataset.
                    else:
                        #Here, we are performing per sample normalisation with values we compute ourselves, and therefore we must compute the normalisation values,
                        #independently for each sample.
                        
                        #We only want to use the foreground voxels for computing the normalisation values..
                        foreground_mask = ForegroundMask(invert=self.invert)(d[key])[0,...] 
                        #We implement this to fill most of the holes that may occur in the foreground
                        fill_binary_holes = binary_fill_holes(foreground_mask) 
                        mean = torch.masked_select(d[key][0,...], torch.from_numpy(fill_binary_holes).to(device=foreground_mask.device)).mean()
                        std = torch.masked_select(d[key][0, ...], torch.from_numpy(fill_binary_holes).to(device=foreground_mask.device)).std()
                        d[key] = NormalizeIntensity(subtrahend=mean, divisor=std)(d[key])    
                
                    d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d 

        if self.version_param == '5':
            
            #Divided into X-ray/X-ray CT and non X-ray modalities.

            for key in self.key_iterator(d):
                if key == 'image':
                    if self.modality in self.x_ray_modalities:
                        if planner_dict == None:
                            raise ValueError("The planner dictionary should not be a nonetype")
                        d[key] = d[key].clamp(min = planner_dict['percentile_00_5_pix'], max = planner_dict['percentile_99_5_pix'])

                        d[key] = NormalizeIntensity(subtrahend=planner_dict['mean_pix'], divisor=planner_dict['std_pix'])(d[key])
                        #We can operate here with a planner dict val because the normalisation is fixed across the entirety of the dataset
                        intensity_aug_mask_dict = {'foreground_region':torch.ones(d[key].shape[1:]).int(), 'foreground_stats_only': False}
                    else:
                        #Here, we are performing per sample normalisation with values we compute ourselves, and therefore we must compute the normalisation values,
                        #independently for each sample.
                        
                        #We only want to use the foreground voxels for computing the normalisation values..
                        foreground_mask = ForegroundMask(invert=self.invert)(d[key])[0,...] 
                        #We implement this to fill most of the holes that may occur in the foreground
                        fill_binary_holes = binary_fill_holes(foreground_mask) 
                        mean = torch.masked_select(d[key][0,...], torch.from_numpy(fill_binary_holes).to(device=foreground_mask.device)).mean()
                        std = torch.masked_select(d[key][0, ...], torch.from_numpy(fill_binary_holes).to(device=foreground_mask.device)).std()
                        d[key] = NormalizeIntensity(subtrahend=mean, divisor=std)(d[key])    
                        #We pass through a dictionary which contains the foreground region (approximately..) and also that we would only like to use the foreground
                        #region voxel values for computing any summary statistics required for any intensity augmentations.
                        intensity_aug_mask_dict = {'foreground_region':torch.from_numpy(fill_binary_holes).to(device=foreground_mask.device).int(), 'foreground_stats_only':True}
                    
                    d[self.foreground_info_key] = intensity_aug_mask_dict
                    # d["preprocessing_original_size"] = [j for i,j in enumerate(d[key].shape) if i != 0]
                else:
                    raise KeyError('The insert key is not supported for this function')
            return d  

if __name__ == '__main__':

    from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, LoadImaged, DivisiblePadd
    import nibabel as nib 

    input_dict = {'image': '/home/parhomesmaeili/DeepEditPlusPlus Development/DeepEditPlusPlus/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTr/BraTS2021_00116.nii.gz'}


    load_stack_minmax_norm = [LoadImaged(keys=("image"), reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=("image")), 
                Orientationd(keys=("image"), axcodes="RAS"), 
                ImageNormalisationd(keys="image", modality="MRI", version_param='1')]

    load_stack_quantile_norm = [LoadImaged(keys=("image"), reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=("image")), 
                Orientationd(keys=("image"), axcodes="RAS"), 
                ImageNormalisationd(keys="image", modality="MRI", version_param='2')]

    load_stack_zscore_foreground = [LoadImaged(keys=("image"), reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=("image")), 
                Orientationd(keys=("image"), axcodes="RAS"), 
                ImageNormalisationd(keys="image", modality="MRI", version_param='5')]
    

    output_minmax = Compose(load_stack_minmax_norm)(input_dict)
    output_quantile = Compose(load_stack_quantile_norm)(input_dict)
    output_zscore = Compose(load_stack_zscore_foreground)(input_dict)

    print('fin!')