'''
File which contains the function which produces the compose list for the get click transform component in the train setup. 
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


def run_get_click_transf(self_dict, context, func_version_param):

    assert type(self_dict) == dict 
    assert type(func_version_param) == str 
    
    supported_version_params = ['1', '2']
    
    assert func_version_param in supported_version_params, 'The version parameter was not supported for the get click transform list composition'

    if func_version_param == '1':
            
        return [
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            #Temporary measure, not the final resolution for the issue with this code deleting the meta dictionary.
            ExtractMetad(keys=("image"), version_param='1'),
            ToNumpyd(keys=("image", "label", "pred")),
            # Transforms for click simulation
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy", version_param='0'),
            AddRandomGuidanceDeepEditd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
                version_param='1'
            ),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            #
            AddSegmentationInputChannelsd(keys=["image"], previous_seg_name = "pred", number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag=True, version_param='1'),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names", "saved_meta")),
        ]
    
    if func_version_param == '2':
        #Marginal change from version 1, it now passes the click set and prediction through to the inner loop again. 
        return [
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            #Temporary measure, not the final resolution for the issue with this code deleting the meta dictionary.
            ExtractMetad(keys=("image"), version_param='1'),
            ToNumpyd(keys=("image", "label", "pred")),
            # Transforms for click simulation
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy", version_param='0'),
            AddRandomGuidanceDeepEditd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
                version_param='1'
            ),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            #
            AddSegmentationInputChannelsd(keys=["image"], previous_seg_name = "pred", number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag=True, version_param='1'),
            ToTensord(keys=("image", "label","pred")),
            SelectItemsd(keys=("image", "label", "label_names", "pred", "saved_meta", "guidance")),
        ]
    
