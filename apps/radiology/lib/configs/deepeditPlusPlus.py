# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = os.path.join(up(up(os.path.abspath(__file__))), 'app_utils', 'deepeditplusplus_utils')
sys.path.append(deepeditpp_utils_dir) 

import logging
import os
from typing import Any, Dict, Optional, Union

# import lib.infers
# import lib.trainers
# from monai.networks.nets import UNETR, DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

####################################################################### 
# 
# External Validation metric imports 


import csv
import shutil
import json
import datetime 
import copy 

##########################################################################

#Imports from the utils (for parametrisation of the config file):

config_utils_dir = os.path.join(deepeditpp_utils_dir, 'config_utils')
sys.path.append(config_utils_dir) 

from config_utils.network_config_utils import run_get_network_configs 
from config_utils.scoring_config_utils import run_scoring_method
from config_utils.a_l_strategy_config_utils import run_strategy_method 
from config_utils.infer_config_utils import run_infer_class_config 
from config_utils.train_config_utils import run_train_class_config 

######################################################################
###############################
logger = logging.getLogger(__name__)


class DeepEditPlusPlus(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        '''
        Config file which sets up the train and inference setup scripts. 

        Inputs: 
        Name: Name of the model?
        Model directory: The upper most level model directory which will contain every folder containing a training run's sets of model parameters/information.
        Conf: The configuration dictionary used to configure the train/inference setup scripts/contains all of the information used for these scripts.
        Planner: The heuristic planner used to extract information from the data and the machine specs.

        '''

        ################# Setting up the imports for the version parameters being used for the train/infer components #########################################

        #Split it by the mode, whether it is for train or for inference. This is incase there are disparities between the two for instances where variable names
        #may be the same.

        if self.conf.get('config_mode') == 'train':

            #For train, set the version parameters of the components in train_setup in this manner. 

            train_version_params = dict() 

            train_version_params["optimizer_version_param"] = self.conf.get("optimizer_version_param")
            train_version_params["loss_func_version_param"] = self.conf.get("loss_func_version_param")
            train_version_params["get_click_version_param"] = self.conf.get("get_click_version_param")
            train_version_params["train_pre_transforms_version_param"] = self.conf.get("train_pre_transforms_version_param")
            train_version_params["train_post_transforms_version_param"] = self.conf.get("train_post_transforms_version_param")
            train_version_params["val_pre_transforms_version_param"] = self.conf.get("val_pre_transforms_version_param")
            train_version_params["val_post_transforms_version_param"] = self.conf.get("val_post_transforms_version_param")
            train_version_params["train_inferer_version_param"] = self.conf.get("train_inferer_version_param")
            train_version_params["val_inferer_version_param"] = self.conf.get("val_inferer_version_param")
            train_version_params["train_iter_update_version_param"] = self.conf.get("train_iter_update_version_param")
            train_version_params["val_iter_update_version_param"] = self.conf.get("val_iter_update_version_param")
            train_version_params["train_key_metric_version_param"] = self.conf.get("train_key_metric_version_param")
            train_version_params["val_key_metric_version_param"] = self.conf.get("val_key_metric_version_param")
            train_version_params["train_handler_version_param"] = self.conf.get("train_handler_version_param")
            train_version_params["engine_version_param"] = self.conf.get("engine_version_param")

            self.train_version_params = train_version_params 
            self.network_version_param = self.conf.get("network_version_param")

            self.train_config_version_param = self.conf.get("train_config_version_param")

        
        elif self.conf.get('config_mode') == 'infer':

            #For inference, set the version parameters in inference_setup in this manner. 

            infer_version_params = dict() 

            infer_version_params['pre_transforms_version_param'] = self.conf.get("pre_transforms_version_param")
            infer_version_params['inverse_transforms_version_param'] = self.conf.get("inverse_transforms_version_param")
            infer_version_params['post_transforms_version_param'] = self.conf.get("post_transforms_version_param")
            infer_version_params['inferer_version_param'] = self.conf.get("inferer_version_param")

            self.infer_version_params = infer_version_params 

            self.network_version_param = self.conf.get("network_version_param")
            
            self.infer_config_version_param = self.conf.get("infer_config_version_param")

        ################################################################################################

        #Setting up the set of version param imports required for the active learning/scoring methods. Hardcoding this for now.

        self.strategy_method_version_param = self.conf.get("strategy_method_version_param")
        self.scoring_method_version_param = self.conf.get("scoring_method_version_param")

        ############################################################################################################################




        ###################### Setting the location to extract the class label configs and dataset configs from ####################
        codebase_dir_name = up(up(up(up(up(__file__)))))
        
        label_config_path = os.path.join(codebase_dir_name, 'datasets', self.conf.get('dataset_name'), 'label_configs.txt')
        
        dataset_json_path = os.path.join(codebase_dir_name, 'datasets', self.conf.get('dataset_name'), 'dataset.json')

        ################### Importing the label configs dictionary #####################

        with open(label_config_path) as f:
            config_dict = json.load(f)
        with open(dataset_json_path) as f:
            dataset_config_dict = json.load(f)

        self.labels = config_dict["labels"]

        self.imaging_modality = dataset_config_dict['modality']

        ###################################################################################

        '''Setting the number of intensity channels (1 we will always be working with unimodal data)'''

        self.number_intensity_ch = 1





        ''' Extracting the information which is used for naming model files/checkpoints/finding checkpoint paths etc. ''' 


        #Adding the datetime for saving the model weights for train, or extracting the datetime set in the inference_main script:

        if self.conf.get("config_mode") == "train":
            self.datetime_now = self.conf.get("datetime")
        elif self.conf.get("config_mode") == "infer":
            self.datetime_now = self.conf.get("datetime")

        ################ Extracting parameters which dicate how many epochs/interval size for saving checkpoints ####################
        
        self.max_epochs = self.conf.get("max_epochs", 50)
        self.save_interval = self.conf.get("save_interval", 10)
        
        # Models directory name which contains the specific datetime's checkpoints etc.:

        #We may wish to extract a checkpoint of parameters.. 

        model_checkpoint = self.conf.get("checkpoint", None) #If a specific checkpoint/save is being used, else use the default

        self.model_checkpoints_folder = 'models'
                        
        # Model checkpoint for train/inference : 

        if self.conf.get("config_mode") == 'train':
            #In this case, the model checkpoint being used is not required.

            # if model_checkpoint == None:
            model_weights_path = os.path.join(self.model_dir, self.datetime_now, self.model_checkpoints_folder + '.pt')

            #If re-starting, then do so from "best" TODO: possibly needs to be changed at a latter point when we do active learning^ so we can do from specific checkpoints instead possibly.

            self.train_paths = [
            os.path.join(self.model_dir, self.datetime_now, "models.pt"), # pretrained SHOULD NOT EXIST PRIOR TO TRAINING until we have an actively train/deploy model
            model_weights_path  # For training, this is the path for the "best" checkpoint. 
            ]

        elif self.conf.get("config_mode") == 'infer':
            
            #If there is no model checkpoint provided, we are just using the checkpoint with the "best" validation value as computed directly according to the
            #validation parametrisation configurations for validation/inner loop etc.

            if model_checkpoint == None: #Just use the "best"
                model_weights_path = os.path.join(self.model_dir, self.datetime_now, self.model_checkpoints_folder + '.pt')
                model_checkpoint = 'best_val_score_epoch'
            
            #If there is a checkpoint provided, then we use that one. 
            else:
                model_weights_path = os.path.join(self.model_dir, self.datetime_now, self.model_checkpoints_folder, 'train_01', model_checkpoint + '.pt')
                
            self.infer_path = [
            model_weights_path  #For inference, this is the path to the checkpoint weights being used.
            ]
            
    


        #### Download PreTrained Model #### 

        #We will turn this off. 
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            # url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            # url = f"{url}/radiology_deepedit_{network}_multilabel.pt"
            # download_file(url, self.path[0])
            pass 
        
        ### Extracting specific variables that specifically pertain to the definitions of the train_setup configuration####
         
        if self.conf.get("config_mode") == "train":

            try:
                self.target_spacing = json.loads(self.conf.get('target_spacing'))  # target image spacing
            except:
                self.target_spacing = None

            try:
                self.spatial_size = json.loads(self.conf.get('spatial_size'))  # train input size
            except:
                self.spatial_size = None

            try:
                self.divisible_padding_factor = json.loads(self.conf.get('divisible_padding_factor'))
            except:
                self.divisible_padding_factor = None 

            try: 
                self.max_iterations = int(self.conf.get('max_iterations'))
            except:
                self.max_iterations = None

            ###

            interactive_init_prob_train = self.conf.get("interactive_init_prob_train", None)

            try: 
                num, den = interactive_init_prob_train.split('/')
                self.interactive_init_prob_train = float(num)/float(den)
            except: 
                num = interactive_init_prob_train 
                self.interactive_init_prob_train = float(num)

            deepedit_prob_train = self.conf.get("deepedit_prob_train", None)

            try: 
                num, den = deepedit_prob_train.split('/')
                self.deepedit_prob_train = float(num)/float(den)
            except: 
                num = deepedit_prob_train 
                self.deepedit_prob_train = float(num)

            interactive_init_prob_val = self.conf.get("interactive_init_prob_val", None)
            try: 
                num, den = interactive_init_prob_val.split('/')
                self.interactive_init_prob_val = float(num)/float(den)
            except: 
                num = interactive_init_prob_val 
                self.interactive_init_prob_val = float(num)

            deepedit_prob_val =  self.conf.get("deepedit_prob_val", None)
            try: 
                num, den = deepedit_prob_val.split('/')
                self.deepedit_prob_val = float(num)/float(den)
            except: 
                num = deepedit_prob_val 
                self.deepedit_prob_val = float(num)

        ###### Extracting specific variables that specifically pertain to the definitions of the inference_setup.py configuration ####

        elif self.conf.get("config_mode") == "infer":

            try:
                self.target_spacing = json.loads(self.conf.get('target_spacing', None))  # target image spacing
            except:
                self.target_spacing = None 

            try:
                self.spatial_size = json.loads(self.conf.get('spatial_size', None))  # train input size
            except:
                self.spatial_size = None 
            
            try:
                self.divisible_padding_factor = json.loads(self.conf.get('divisible_padding_factor', None))
            except:
                self.divisible_padding_factor = None 


        ''' Extracting variables pertaining to the definitions of the A.L strategy setup '''
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")


        # Network
        
        self.networks_dict = run_get_network_configs(self.network_version_param, dict(vars(self)))
        assert type(self.networks_dict) == dict, "The network configurations were not provided in a dict format"





        #Dumping all of the configuration parameters into a file in the model folder for training:


        if self.conf.get("config_mode") == "train":

            ###### Name of the text file which we are going to save our train configs info to ##################

            train_configs_filename = 'train_config_settings.txt'
            
            #create a duplicate dict:
            duplicate_conf = copy.deepcopy(self.conf)
            # del duplicate_conf["mode"]
            
            os.makedirs(os.path.join(self.model_dir, self.datetime_now))
            with open(os.path.join(self.model_dir, self.datetime_now, train_configs_filename), 'w') as file:
                file.write(json.dumps(duplicate_conf,indent=2)) 

        
            ##################### EXTERNAL VALIDATION SAVES (metrics + images optionally) #####################################################
            self.external_validation_output_dir = os.path.join(os.path.abspath(codebase_dir_name), 'external_validation', self.datetime_now, self.model_checkpoints_folder)
            
            output_val_dir_scores = os.path.join(self.external_validation_output_dir, 'validation_scores')
            # output_dir_images = os.path.join(self.external_validation_output_dir, 'validation_images_verif')

            run_mode = self.conf.get("mode", "train")

            if run_mode == "train":
                if os.path.exists(self.external_validation_output_dir):
                    shutil.rmtree(self.external_validation_output_dir)

                os.makedirs(output_val_dir_scores)
                # os.makedirs(output_dir_images)


                fields = ['interactive_init_dice', 'autoseg_dice', 'deepedit_val_config_dice']    
                with open(os.path.join(output_val_dir_scores, 'validation.csv'),'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields) 
            
            ################################################################################################################################################
            #

            #Dumping all of the training configuration parameters into the external validation folder also.

            
            #create a duplicate dict without the mode name:
            duplicate_conf = copy.deepcopy(self.conf)
            # del duplicate_conf["mode"]
            
            with open(os.path.join(self.external_validation_output_dir, train_configs_filename), 'w') as file:
                file.write(json.dumps(duplicate_conf,indent=2)) 

        elif self.conf.get("config_mode") == "infer":
            
            ###### Name of the text file which we are going to save our infer configs info to ##################

            infer_configs_filename = 'infer_config_settings.txt'
            infer_run_name = self.conf.get("infer_run_name")
            infer_run_num = self.conf.get("infer_run_num")

            infer_run_click_parametrisation_str  = self.conf.get("infer_click_parametrisations_string")
            assert infer_run_click_parametrisation_str != None

            #create a duplicate dict:
            duplicate_conf = copy.deepcopy(self.conf)
            # del duplicate_conf["mode"]
            
            infer_configs_folder = os.path.join(codebase_dir_name, 'datasets', self.conf.get('dataset_name'), self.conf.get('infer_type') + f"_{self.conf.get('simulation_type')}", self.datetime_now, model_checkpoint, infer_run_name, infer_run_click_parametrisation_str, 'run_' + infer_run_num + '_infer_configs')

            if os.path.exists(infer_configs_folder):
                shutil.rmtree(infer_configs_folder)
            
            os.makedirs(infer_configs_folder)

            with open(os.path.join(infer_configs_folder, infer_configs_filename), 'w') as file:
                file.write(json.dumps(duplicate_conf,indent=2)) 


    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        
        return run_infer_class_config(self.infer_config_version_param, dict(vars(self)))

    def trainer(self) -> Optional[TrainTask]:
        
        return run_train_class_config(self.train_config_version_param, dict(vars(self)))

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        
        return run_strategy_method(self.strategy_method_version_param, dict(vars(self)))

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        
        return run_scoring_method(self.scoring_method_version_param, self.networks_dict, InferDeepEditPlusPlus, dict(vars(self)))
        