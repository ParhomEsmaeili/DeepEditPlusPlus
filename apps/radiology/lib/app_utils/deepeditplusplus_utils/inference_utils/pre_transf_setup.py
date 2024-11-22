'''Pre-transforms for performing inference. 

Supported version:

Version -1: Configured the same as version 2, but only intended for a fully autoseg model.


Version 1: DeepEdit++ v1.1 from the upgrade report, which was intended to be implemented with padding for min-max normalised images. 

Version 2: Modification from min-max normalisation, to per-image quartile based normalisation in the non-CT modalities.

Version: 
'''

from monai.transforms import (
    # Activationsd,
    # AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    # Resized,
    # ScaleIntensityRanged,
    # SqueezeDimd,
    ToNumpyd,
    DivisiblePadd,
    # CenterSpatialCropd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType

##################################################

#Imports for the parametrised utils

from os.path import dirname as up
import sys
import os

deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from transforms_utils.add_guidance_from_3dpointsd import AddGuidanceFromPointsDeepEditd
from transforms_utils.add_guidance_signald import AddGuidanceSignalDeepEditd
from transforms_utils.discard_add_guidanced import DiscardAddGuidanced
from transforms_utils.resize_guidance_multilabeld import ResizeGuidanceMultipleLabelDeepEditd
from transforms_utils.add_segmentation_input_channelsd import AddSegmentationInputChannelsd
from transforms_utils.mapping_guidance_pointsd import MappingGuidancePointsd
from transforms_utils.modality_based_normalisationd import ImageNormalisationd
from inference_transforms_utils.get_original_img_infod import GetOriginalInformationd

def run_get_inference_pre_transf(self_dict, data, func_version_param):
    
    if func_version_param == '-1':

        if self_dict["type"] == InferType.SEGMENTATION:
            t = [
            LoadImaged(keys="image", reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            GetOriginalInformationd(keys=["image"], version_param='0'),
            ImageNormalisationd(keys="image", modality=self_dict["modality"], version_param='2'),
            DivisiblePadd(keys=("image"), k=self_dict["transforms_parametrisation_dict"]["divisible_padding_factor"]),
            ]
            
        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t
        
    elif func_version_param == '1':

        if self_dict["type"] == InferType.DEEPEDIT:
            t = [
                LoadImaged(keys=["image", "previous_seg"], reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=["image", "previous_seg"]),
                Orientationd(keys=["image", "previous_seg"], axcodes="RAS"),
                GetOriginalInformationd(keys=["image"], version_param='0'),
                ImageNormalisationd(keys="image", modality=self_dict["modality"], version_param='1'),
                DivisiblePadd(keys=("image", "previous_seg"), k=self_dict["transforms_parametrisation_dict"]["divisible_padding_factor"]),
                #NOTE: The preprocessing original size is only placed into the data dict from the image normalisation transform! 
                MappingGuidancePointsd(keys=("image"), original_spatial_size_key="original_size", label_names=self_dict["labels"], version_param='1'), 
                AddGuidanceSignalDeepEditd(
                    keys="image", guidance="guidance", number_intensity_ch=self_dict["number_intensity_ch"], label_names = self_dict["labels"], version_param='1'
                ),
                AddSegmentationInputChannelsd(keys=["image"], previous_seg_name="previous_seg", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= True, version_param='1')
            ]
            
        else:
            t = [
            LoadImaged(keys="image", reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            GetOriginalInformationd(keys=["image"], version_param='0'),
            ImageNormalisationd(keys="image", modality=self_dict["modality"], version_param='1'),
            DivisiblePadd(keys=("image"), k=self_dict["transforms_parametrisation_dict"]["divisible_padding_factor"]),
            ]
            if self_dict["type"] == InferType.DEEPGROW:
            
                t.extend(
                    [   
                        #Note here that the preprocessing original size param in the data dict comes from the image normalisation transform
                        MappingGuidancePointsd(keys="image", original_spatial_size_key="original_size", label_names=self_dict["labels"], version_param='1'),
                        AddGuidanceSignalDeepEditd(
                            keys="image", guidance="guidance", number_intensity_ch=self_dict["number_intensity_ch"], label_names = self_dict["labels"], version_param='1'
                        ),
                        AddSegmentationInputChannelsd(keys="image", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= False, version_param='1')
                    ]
                )
            
            elif self_dict["type"] == InferType.SEGMENTATION:
                t.extend(
                    [
                        DiscardAddGuidanced(
                            keys="image", label_names=self_dict["labels"], number_intensity_ch=self_dict["number_intensity_ch"], version_param='1'),
                        AddSegmentationInputChannelsd(keys="image", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= False, version_param='1'),
                    ]
                )
        
        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    elif func_version_param == '2': 

        if self_dict["type"] == InferType.DEEPEDIT:
            t = [
                LoadImaged(keys=["image", "previous_seg"], reader="ITKReader", image_only=False), 
                EnsureChannelFirstd(keys=["image", "previous_seg"]),
                Orientationd(keys=["image", "previous_seg"], axcodes="RAS"),
                GetOriginalInformationd(keys=["image"], version_param='0'),
                ImageNormalisationd(keys="image", modality=self_dict["modality"], version_param='2'),
                DivisiblePadd(keys=("image", "previous_seg"), k=self_dict["transforms_parametrisation_dict"]["divisible_padding_factor"]),
                #NOTE: The preprocessing original size is only placed into the data dict from the image normalisation transform! 
                MappingGuidancePointsd(keys=("image"), original_spatial_size_key="original_size", label_names=self_dict["labels"], version_param='1'), 
                AddGuidanceSignalDeepEditd(
                    keys="image", guidance="guidance", number_intensity_ch=self_dict["number_intensity_ch"], label_names = self_dict["labels"], version_param='1'
                ),
                AddSegmentationInputChannelsd(keys=["image"], previous_seg_name="previous_seg", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= True, version_param='1')
            ]
            
        else:
            t = [
            LoadImaged(keys="image", reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            GetOriginalInformationd(keys=["image"], version_param='0'),
            ImageNormalisationd(keys="image", modality=self_dict["modality"], version_param='2'),
            DivisiblePadd(keys=("image"), k=self_dict["transforms_parametrisation_dict"]["divisible_padding_factor"]),
            ]
            if self_dict["type"] == InferType.DEEPGROW:
            
                t.extend(
                    [   
                        #Note here that the preprocessing original size param in the data dict comes from the image normalisation transform
                        MappingGuidancePointsd(keys="image", original_spatial_size_key="original_size", label_names=self_dict["labels"], version_param='1'),
                        AddGuidanceSignalDeepEditd(
                            keys="image", guidance="guidance", number_intensity_ch=self_dict["number_intensity_ch"], label_names = self_dict["labels"], version_param='1'
                        ),
                        AddSegmentationInputChannelsd(keys="image", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= False, version_param='1')
                    ]
                )
            
            elif self_dict["type"] == InferType.SEGMENTATION:
                t.extend(
                    [
                        DiscardAddGuidanced(
                            keys="image", label_names=self_dict["labels"], number_intensity_ch=self_dict["number_intensity_ch"], version_param='1'),
                        AddSegmentationInputChannelsd(keys="image", number_intensity_ch = self_dict["number_intensity_ch"], label_names=self_dict["labels"], previous_seg_flag= False, version_param='1'),
                    ]
                )
        
        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t
