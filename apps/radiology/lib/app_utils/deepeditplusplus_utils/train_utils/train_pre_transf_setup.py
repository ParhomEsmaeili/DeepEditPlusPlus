'''
File which contains the function which produces the compose list for the pre transform component in the train setup. 
'''

import logging

import torch

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from transforms_utils.add_guidance_signald import AddGuidanceSignalDeepEditd
from transforms_utils.add_initial_seed_point_missing_labelsd import AddInitialSeedPointMissingLabelsd
from transforms_utils.add_random_guidanced import AddRandomGuidanceDeepEditd
from transforms_utils.find_all_valid_slices_missinglabelsd import FindAllValidSlicesMissingLabelsd
from transforms_utils.find_discrepancy_regionsd import FindDiscrepancyRegionsDeepEditd
from transforms_utils.splitpredlabelsd import SplitPredsLabeld
from transforms_utils.normalize_labels_in_datasetd import NormalizeLabelsInDatasetd
from transforms_utils.add_segmentation_input_channelsd import AddSegmentationInputChannelsd
from transforms_utils.extract_meta_informationd import ExtractMetad
from transforms_utils.mapping_labels_from_configd import MappingLabelsInDatasetd
from transforms_utils.modality_based_normalisationd import ImageNormalisationd

# inner_loop_utils.inner_loop_version_1 import run as inner_loop_1_run

#from lib.transforms.transforms import NormalizeLabelsInDatasetd
# from monailabel.deepeditPlusPlus.interaction import Interaction
# from monailabel.deepeditPlusPlus.transforms import (
#     AddGuidanceSignalDeepEditd,
#     AddInitialSeedPointMissingLabelsd,
#     AddRandomGuidanceDeepEditd,
#     FindAllValidSlicesMissingLabelsd,
#     FindDiscrepancyRegionsDeepEditd,
#     SplitPredsLabeld,
#     NormalizeLabelsInDatasetd,
#     AddSegmentationInputChannels,
#     #ExtractChannelsd,
#     MappingLabelsInDatasetd,
#     ExtractMeta,
#     IntensityCorrection,
# )


from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    SelectItemsd,
    ToNumpyd,
    ToTensord,
    ToDeviced,
    DivisiblePadd,
    CenterSpatialCropd
)

logger = logging.getLogger(__name__)

def run_get_train_pre_transf(self_dict, context, func_version_param):

    assert type(self_dict) == dict 
    assert type(func_version_param) == str 

    supported_version_params = ['1','2', '3']
    
    assert func_version_param in supported_version_params, 'The version parameter was not supported for the get train pre-transform list composition'

    if func_version_param == '1':
            
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='1'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]
    
    elif func_version_param == '2':

        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='2'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]
    
    elif func_version_param == '3':
        #Modified version of version_2 but with the added modification that the click set is propagated to the inner loop again. 
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='2'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "guidance", "label_names")),
        ]
