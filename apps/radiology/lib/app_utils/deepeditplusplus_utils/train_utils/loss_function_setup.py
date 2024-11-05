'''
File which contains the function which produces the compose list for the loss function component in the train setup. 
'''

import logging

import torch

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from monai.losses import DiceCELoss 

# from deepeditplusplus_utils.transforms_utils.add_guidance_signald import AddGuidanceSignalDeepEditd
# from deepeditplusplus_utils.transforms_utils.add_initial_seed_point_missing_labelsd import AddInitialSeedPointMissingLabelsd
# from deepeditplusplus_utils.transforms_utils.add_random_guidanced import AddRandomGuidanceDeepEditd
# from deepeditplusplus_utils.transforms_utils.find_all_valid_slices_missinglabelsd import FindAllValidSlicesMissingLabelsd
# from deepeditplusplus_utils.transforms_utils.find_discrepancy_regionsd import FindDiscrepancyRegionsDeepEditd
# from deepeditplusplus_utils.transforms_utils.splitpredlabelsd import SplitPredsLabeld
# from deepeditplusplus_utils.transforms_utils.normalize_labels_in_datasetd import NormalizeLabelsInDatasetd
# from deepeditplusplus_utils.transforms_utils.add_segmentation_input_channelsd import AddSegmentationInputChannelsd
# from deepeditplusplus_utils.transforms_utils.extract_meta_informationd import ExtractMetad
# from deepeditplusplus_utils.transforms_utils.mapping_labels_from_configd import MappingLabelsInDatasetd
# from deepeditplusplus_utils.transforms_utils.modality_based_normalisationd import ImageNormalisationd


# from monai.transforms import (
#     Activationsd,
#     AsDiscreted,
#     EnsureChannelFirstd,
#     LoadImaged,
#     Orientationd,
#     RandFlipd,
#     RandRotate90d,
#     RandShiftIntensityd,
#     Resized,
#     ScaleIntensityRanged,
#     SelectItemsd,
#     ToNumpyd,
#     ToTensord,
#     ToDeviced,
#     DivisiblePadd,
#     CenterSpatialCropd
# )

logger = logging.getLogger(__name__)

def run_get_loss_func(self_dict, context, func_version_param):

    assert type(self_dict) == dict 
    assert type(func_version_param) == str 

    supported_version_params = ['0']
    
    assert func_version_param in supported_version_params, 'The version parameter was not supported for obtaining the loss function'

    if func_version_param == '0':

        return DiceCELoss(to_onehot_y=True, softmax=True)

