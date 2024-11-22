import logging
import os
from typing import Any, Dict, Optional, Union

# import lib.infers
# import lib.trainers
# from monai.networks.nets import UNETR, DynUNet

# from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
# from monailabel.interfaces.tasks.scoring import ScoringMethod
# from monailabel.interfaces.tasks.strategy import Strategy
# from monailabel.interfaces.tasks.train import TrainTask
# from monailabel.tasks.activelearning.epistemic import Epistemic
# from monailabel.tasks.scoring.dice import Dice
# from monailabel.tasks.scoring.epistemic import EpistemicScoring
# from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

####################################################################### 
# 
# External Validation metric imports 

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir) 

from inference_utils.inference_setup import DeepEditPlusPlus as InferDeepEditPlusPlus

import csv
import shutil
import json
import datetime 
import copy 

logger = logging.getLogger(__name__)

'''
Version -1: The fully autoseg only network's config (i.e. only uses the image!).
Version 1: The deepedit++ v1.1 like config
'''

def run_infer_class_config(version_param, self_dict):
        
    supported_versions = ['-1', '1']
    assert version_param in supported_versions, 'The infer class config setup was not supported' 

    if version_param == '-1':
        
        print('checkpoint path is:')
        print(self_dict['infer_path'])
        
        transforms_params_dict = dict()  #Contains the information about permutable variables that parametrise the transforms..

        transforms_params_dict['spatial_size'] = self_dict['spatial_size']
        transforms_params_dict['target_spacing'] = self_dict['target_spacing']
        transforms_params_dict['divisible_padding_factor'] = self_dict['divisible_padding_factor']

        return {
            
            f"{self_dict['name']}_autoseg": InferDeepEditPlusPlus(
                path=self_dict['infer_path'],
                modality=self_dict['imaging_modality'],
                infer_version_params = self_dict['infer_version_params'],
                transforms_parametrisation_dict = transforms_params_dict, 
                network=self_dict['networks_dict']['base_network'],
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self_dict['labels'],
                preload=strtobool(self_dict['conf'].get("preload", "false")),
                number_intensity_ch=self_dict['number_intensity_ch'],
                type=InferType.SEGMENTATION,
            )
        }
    

    elif version_param == '1':
        
        print('checkpoint path is:')
        print(self_dict['infer_path'])
        
        transforms_params_dict = dict()  #Contains the information about permutable variables that parametrise the transforms..

        transforms_params_dict['spatial_size'] = self_dict['spatial_size']
        transforms_params_dict['target_spacing'] = self_dict['target_spacing']
        transforms_params_dict['divisible_padding_factor'] = self_dict['divisible_padding_factor']

        return {
            self_dict['name']: InferDeepEditPlusPlus(
                path=self_dict['infer_path'],
                modality=self_dict['imaging_modality'],
                infer_version_params = self_dict['infer_version_params'],
                transforms_parametrisation_dict = transforms_params_dict, 
                network=self_dict['networks_dict']['base_network'],
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self_dict['labels'],
                preload=strtobool(self_dict['conf'].get("preload", "false")),
                number_intensity_ch=self_dict['number_intensity_ch'],
                config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
            f"{self_dict['name']}_autoseg": InferDeepEditPlusPlus(
                path=self_dict['infer_path'],
                modality=self_dict['imaging_modality'],
                infer_version_params = self_dict['infer_version_params'],
                transforms_parametrisation_dict = transforms_params_dict, 
                network=self_dict['networks_dict']['base_network'],
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self_dict['labels'],
                preload=strtobool(self_dict['conf'].get("preload", "false")),
                number_intensity_ch=self_dict['number_intensity_ch'],
                type=InferType.SEGMENTATION,
            ),
            f"{self_dict['name']}_interactive_init": InferDeepEditPlusPlus(
                path=self_dict['infer_path'],
                modality=self_dict['imaging_modality'],
                infer_version_params = self_dict['infer_version_params'],
                transforms_parametrisation_dict = transforms_params_dict, 
                network=self_dict['networks_dict']['base_network'],
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self_dict['labels'],
                preload=strtobool(self_dict['conf'].get("preload","false")),
                number_intensity_ch=self_dict['number_intensity_ch'],
                type=InferType.DEEPGROW,
            )
        }
