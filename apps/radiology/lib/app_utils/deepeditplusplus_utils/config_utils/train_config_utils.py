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
from monailabel.interfaces.tasks.train import TrainTask
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

from train_utils.train_setup import DeepEditPlusPlus as TrainDeepEditPlusPlus

import csv
import shutil
import json
import datetime 
import copy 

logger = logging.getLogger(__name__)


def run_train_class_config(version_param, self_dict):

    supported_versions = ['1']

    assert version_param in supported_versions

    if version_param == '1':

    
        output_dir = os.path.join(self_dict['model_dir'], self_dict['datetime_now'], self_dict['model_checkpoints_folder']) 
        load_path = self_dict['train_paths'][0] if os.path.exists(self_dict['train_paths'][0]) else self_dict['train_paths'][1]

        #Producing the dictionary which contains the information about the parametrisation of the transforms and inner loop within the transform components! 
        # (E.g. spatial size, probability for autoseg init, padding size etc)
        component_parametrisation_dict = dict() 

        component_parametrisation_dict['spatial_size'] = self_dict['spatial_size']
        component_parametrisation_dict['target_spacing'] = self_dict['target_spacing']
        component_parametrisation_dict['divisible_padding_factor'] = self_dict['divisible_padding_factor']
        component_parametrisation_dict['init_lr'] = self_dict['init_lr']
        component_parametrisation_dict['max_iterations'] = self_dict['max_iterations']
        component_parametrisation_dict['interactive_init_probability_train'] = self_dict['interactive_init_prob_train']
        component_parametrisation_dict['deepedit_probability_train'] = self_dict['deepedit_prob_train'] 
        component_parametrisation_dict['interactive_init_probability_val'] = self_dict['interactive_init_prob_val'] 
        component_parametrisation_dict['deepedit_probability_val'] = self_dict['deepedit_prob_val'] 

        task: TrainTask = TrainDeepEditPlusPlus(
            model_dir=output_dir,
            network=self_dict['networks_dict']['train_network'],
            labels=self_dict['labels'],
            train_version_params = self_dict['train_version_params'],
            component_parametrisation_dict = component_parametrisation_dict,
            modality = self_dict['imaging_modality'],
            external_validation_dir=self_dict['external_validation_output_dir'], 
            n_saved=int(self_dict['max_epochs']) // int(self_dict['save_interval']),
            load_path=load_path,
            publish_path=self_dict['train_paths'][1],
            number_intensity_ch=self_dict['number_intensity_ch'],
            config={"pretrained": strtobool(self_dict['conf'].get("use_pretrained_model", "true"))},
            debug_mode=False, #True        
            find_unused_parameters=True,
        )
        return task
