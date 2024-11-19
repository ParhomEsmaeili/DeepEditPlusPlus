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

import json
import logging
import os
import shutil
from typing import Dict
############################
from os.path import dirname as up
import os
import sys
file_dir = up(up(up(os.path.abspath(__file__))))
sys.path.append(file_dir)
print(sys.path)
import torch 
import re
########################

import lib.configs
from lib.activelearning import Last
# from lib.infers.deepgrow_pipeline import InferDeepgrowPipeline
# from lib.infers.vertebra_pipeline import InferVertebraPipeline


import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
# from monailabel.scribbles.infer import GMMBasedGraphCut, HistogramBasedGraphCut
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random

# bundle
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import get_bundle_models, strtobool
from monailabel.utils.others.planner import HeuristicPlanner

logger = logging.getLogger(__name__)

########### Additional modules for the inference and training scripts ############
import shutil
from monailabel.utils.others.generic import device_list, file_ext
import nibabel as nib 
import numpy as np
import ast



########## Imports for simulations.

deepeditpp_runscript_utils_dir = os.path.join(up(os.path.abspath(__file__)), 'lib', 'app_utils', 'deepeditplusplus_run_script_utils')
sys.path.append(deepeditpp_runscript_utils_dir)

from deepeditplusplus_test_utils.probabilistic_click_simulator import probabilistic_click_simulation
from deepeditplusplus_test_utils.inner_loop_simulator import probabilistic_inner_loop_runner
from deepeditplusplus_test_utils.simulation_transf_parametrisation_config import run_generate_test_config_dict




from monailabel.transform.writer import Writer 

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")
        print('app dir {}'.format(app_dir))
        print('studies {}'.format(studies))
        print('conf {}'.format(conf))
        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}
        
        # Load models from app model implementation, e.g., --conf models <segmentation_spleen>
        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)
           
        models = models.split(",")
        models = [m.strip() for m in models]
        # Can be configured with --conf scribbles false or true
        self.scribbles = conf.get("scribbles", "true") == "true"
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[48, 48, 32]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        # app models
        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner)
        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        # Load models from bundle config files, local or released in Model-Zoo, e.g., --conf bundles <spleen_ct_segmentation>
        self.bundles = get_bundle_models(app_dir, conf, conf_key="bundles") if conf.get("bundles") else None

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Radiology ({monailabel.__version__})",
            description="DeepLearning models for radiology",
            version=monailabel.__version__,
        )

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        if self.heuristic_planner:
            self.planner.run(datastore)
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}

        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        #################################################
        # Bundle Models
        #################################################
        # if self.bundles:
        #     for n, b in self.bundles.items():
        #         i = BundleInferTask(b, self.conf)
        #         logger.info(f"+++ Adding Bundle Inferer:: {n} => {i}")
        #         infers[n] = i

        #################################################
        # Scribbles
        #################################################
        # if self.scribbles:
        #     infers.update(
        #         {
        #             "Histogram+GraphCut": HistogramBasedGraphCut(
        #                 intensity_range=(-300, 200, 0.0, 1.0, True),
        #                 pix_dim=(2.5, 2.5, 5.0),
        #                 lamda=1.0,
        #                 sigma=0.1,
        #                 num_bins=64,
        #                 labels=task_config.labels,
        #             ),
        #             "GMM+GraphCut": GMMBasedGraphCut(
        #                 intensity_range=(-300, 200, 0.0, 1.0, True),
        #                 pix_dim=(2.5, 2.5, 5.0),
        #                 lamda=5.0,
        #                 sigma=0.5,
        #                 num_mixtures=20,
        #                 labels=task_config.labels,
        #             ),
        #         }
        #     )

        #################################################
        # Pipeline based on existing infers
        #################################################
        # if infers.get("deepgrow_2d") and infers.get("deepgrow_3d"):
        #     infers["deepgrow_pipeline"] = InferDeepgrowPipeline(
        #         path=self.models["deepgrow_2d"].path,
        #         network=self.models["deepgrow_2d"].network,
        #         model_3d=infers["deepgrow_3d"],
        #         description="Combines Clara Deepgrow 2D and 3D models",
        #     )

        #################################################
        # # Pipeline based on existing infers for vertebra segmentation
        # # Stages:
        # # 1/ localization spine
        # # 2/ localization vertebra
        # # 3/ segmentation vertebra
        # #################################################
        # if (
        #     infers.get("localization_spine")
        #     and infers.get("localization_vertebra")
        #     and infers.get("segmentation_vertebra")
        # ):
        #     infers["vertebra_pipeline"] = InferVertebraPipeline(
        #         task_loc_spine=infers["localization_spine"],  # first stage
        #         task_loc_vertebra=infers["localization_vertebra"],  # second stage
        #         task_seg_vertebra=infers["segmentation_vertebra"],  # third stage
        #         description="Combines three stage for vertebra segmentation",
        #     )
        logger.info(infers)
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        # if strtobool(self.conf.get("skip_trainers", "false")):
        #     return trainers
        # #################################################
        # # Models
        # #################################################
        # for n, task_config in self.models.items():
        #     t = task_config.trainer()
        #     if not t:
        #         continue

        #     logger.info(f"+++ Adding Trainer:: {n} => {t}")
        #     trainers[n] = t

        # #################################################
        # # Bundle Models
        # #################################################
        # if self.bundles:
        #     for n, b in self.bundles.items():
        #         t = BundleTrainTask(b, self.conf)
        #         if not t or not t.is_valid():
        #             continue

        #         logger.info(f"+++ Adding Bundle Trainer:: {n} => {t}")
        #         trainers[n] = t

        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
            "last": Last(),
        }

        if strtobool(self.conf.get("skip_strategies", "true")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if strtobool(self.conf.get("skip_scoring", "true")):
            return methods

        for n, task_config in self.models.items():
            s = task_config.scoring_method()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Scoring Method:: {k} => {v}")
                methods[k] = v

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods


    #####################################################################


"""
Example to run train/infer/batch infer/scoring task(s) locally without actually running MONAI Label Server

More about the available app methods, please check the interface monailabel/interfaces/app.py

"""


def main():
    import argparse
    # import shutil
    from pathlib import Path

    # from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    base_directory = up(up(up(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised") 
    parser.add_argument("--model", default="deepeditplusplus")
    parser.add_argument("--config_mode", default='infer')
    parser.add_argument("--infer_type", default="validation")#choices=("test", "validation")
    parser.add_argument('--infer_run', default='0')
    parser.add_argument("--infer_run_name", nargs="+", default=["Editing", "Autoseg", "10"], help="The subtask/mode which we want to execute")
    #Possible options include: Autoseg, Interactive, Editing + Either init mode + number of edit iters.
    parser.add_argument("--infer_click_parametrised_bool", default=False) #The bool which contains the information about whether the click is parametrised or not.
    parser.add_argument("--infer_click_parametrisations", nargs="+", default=["None"]) #
    
    #The argument which contains the information about the parametrisation of the click size. First element denotes the type of parametrisation, the subsequent parametrise it
    #If it is dynamic (or NONE) then a one element list containing "Dynamic Click Size", else it should be a list of the dimension-wise parametrisations (INTS) for a fixed parametrisation.     
    
    parser.add_argument("--sequentiality_mode", default='SIM') #The mode which dictates whether the clicks are accumulated or not. SIM = no accumulation.
    parser.add_argument("--simulation_transf_config_parametrisation", default='1') #The version param which determines what the generated config for the transforms for performing click simulation is.

    #The parameter which controls whether we are performing probabilistic or deterministic simulation. 
    parser.add_argument("--simulation_type", default='probabilistic')

    #Set of parametrisations for selecting the appropriate configs:
    parser.add_argument("--infer_config_version_param", default='1')
    parser.add_argument("--network_version_param", default='0')
    parser.add_argument("--strategy_method_version_param", default='0')
    parser.add_argument("--scoring_method_version_param", default='0')
    
    #Set of parametrisations which are used for parametrising the transforms (not the set of list composed transforms, but the actual sub-transforms)
    parser.add_argument("--target_spacing", default='[1,1,1]')
    parser.add_argument("--spatial_size", default='[128,128,128]')
    parser.add_argument("--divisible_padding_factor", default='[64,64,32]')

    #Information regarding the checkpoint and model version (and also the validation fold used for performing inference if it is validation)
    parser.add_argument("--checkpoint")
    parser.add_argument("--datetime", default='20241104_135136') 
    parser.add_argument("--val_fold", default='0', help="The fold which is designated as the validation")

    #################################### 
    
    #Adding the inference setup script parametrisations for the components:
    parser.add_argument("--pre_transforms_version_param", default='2')
    parser.add_argument("--inverse_transforms_version_param", default='0')
    parser.add_argument("--post_transforms_version_param", default='0')
    parser.add_argument("--inferer_version_param", default='0')

    args = parser.parse_args()

    app_dir = up(__file__)
    
    ################ Adding the name of the cuda device used ###############
    cuda_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(cuda_device)
    print(device_name)

    ############ Extracting the name of the inference run ####################

    if len(args.infer_run_name) > 1:
            run_name_string = args.infer_run_name[1].title() + "_initialisation_" + args.infer_run_name[2] + '_edit_iters'
    else:
        run_name_string = args.infer_run_name[0]

    ############# Extracting the name of the inference time click parametrisations ################# 

    if args.infer_click_parametrised_bool: 
         
        #In this case we need to generate the name of the folders which contain the corresponding parametrisation. 

        if args.infer_click_parametrisations[0] == "Dynamic Click Size":

            inference_click_parametrisation_string = "Dynamic Click Size"
            inference_click_parametrisations_dict = {"Dynamic Click Size":[]}
        
        elif args.infer_click_parametrisations[0] == "Fixed Click Size":

            inference_click_parametrisation_string = f"Fixed Click Size {args.infer_click_parametrisations[1:]}"
            inference_click_parametrisations_dict = {"Fixed Click Size": [float(i) for i in args.infer_click_parametrisations[1:]]}
    else:

        inference_click_parametrisation_string = "No Click Param"
        inference_click_parametrisations_dict = {"No Click Param":[]}


    ############### We introduce flexibility about which set of model weights to use for inference for configuration #####################
    
    if args.infer_type == "validation": 
        if args.checkpoint != None:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies,
                "config_mode": args.config_mode,
                "target_spacing":args.target_spacing,
                "spatial_size":args.spatial_size,
                "divisible_padding_factor":args.divisible_padding_factor,
                "checkpoint": args.checkpoint,
                "datetime": args.datetime,
                "inference_set": args.val_fold,
                #Params corresponding to the inference_setup.py script.
                "pre_transforms_version_param": args.pre_transforms_version_param,
                "inferer_version_param": args.inferer_version_param,
                "inverse_transforms_version_param": args.inverse_transforms_version_param,
                "post_transforms_version_param": args.post_transforms_version_param,
                #Params corresponding to the test click simulation for inference.
                "simulation_transf_config_parametrisation": args.simulation_transf_config_parametrisation,
                "infer_click_parametrised_bool": args.infer_click_parametrised_bool,
                "infer_click_parametrisations": args.infer_click_parametrisations,
                "infer_click_parametrisations_string": inference_click_parametrisation_string,
                #Introducing the vrsion params for the config files
                "network_version_param": args.network_version_param,
                "infer_config_version_param": args.infer_config_version_param,
                "strategy_method_version_param": args.strategy_method_version_param,
                "scoring_method_version_param": args.scoring_method_version_param,
                #Introducing name of the actual validation/test + the simulation type probablistic or deterministic click generation.
                "infer_type": args.infer_type,
                "simulation_type": args.simulation_type,
                "infer_run_name":run_name_string,
                "infer_run_num": args.infer_run

            }
        else:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies,
                "config_mode": args.config_mode,
                "target_spacing":args.target_spacing,
                "spatial_size":args.spatial_size,
                "divisible_padding_factor":args.divisible_padding_factor,
                "datetime": args.datetime,
                "inference_set": args.val_fold,
                #Params corresponding to the inference_setup.py script.
                "pre_transforms_version_param": args.pre_transforms_version_param,
                "inferer_version_param": args.inferer_version_param,
                "inverse_transforms_version_param": args.inverse_transforms_version_param,
                "post_transforms_version_param": args.post_transforms_version_param,
                #Params corresponding to the test click simulation for inference.
                "simulation_transf_config_parametrisation": args.simulation_transf_config_parametrisation,
                "infer_click_parametrised_bool": args.infer_click_parametrised_bool,
                "infer_click_parametrisations": args.infer_click_parametrisations,
                "infer_click_parametrisations_string": inference_click_parametrisation_string,
               #Introducing the vrsion params for the config files
                "network_version_param": args.network_version_param,
                "infer_config_version_param": args.infer_config_version_param,
                "strategy_method_version_param": args.strategy_method_version_param,
                "scoring_method_version_param": args.scoring_method_version_param,
                #Introducing name of the actual validation/test + the simulation type probablistic or deterministic click generation.
                "infer_type": args.infer_type,
                "simulation_type": args.simulation_type,
                "infer_run_name":run_name_string,
                "infer_run_num": args.infer_run
            }
    elif args.infer_type == "test":

        if args.checkpoint != None:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies,
                "config_mode": args.config_mode,
                "checkpoint": args.checkpoint,
                "datetime": args.datetime,
                "inference_set": 'imagesTs',
                #Params corresponding to the inference_setup.py script.
                "pre_transforms_version_param": args.pre_transforms_version_param,
                "inferer_version_param": args.inferer_version_param,
                "inverse_transforms_version_param": args.inverse_transforms_version_param,
                "post_transforms_version_param": args.post_transforms_version_param,
                #Params corresponding to the test click simulation for inference.
                "simulation_transf_config_parametrisation": args.simulation_transf_config_parametrisation,
                "infer_click_parametrised_bool": args.infer_click_parametrised_bool,
                "infer_click_parametrisations": args.infer_click_parametrisations,
                "infer_click_parametrisations_string": inference_click_parametrisation_string,
                #Introducing the vrsion params for the config files
                "network_version_param": args.network_version_param,
                "infer_config_version_param": args.infer_config_version_param,
                "strategy_method_version_param": args.strategy_method_version_param,
                "scoring_method_version_param": args.scoring_method_version_param,
                #Introducing name of the actual validation/test + the simulation type probablistic or deterministic click generation.
                "infer_type": args.infer_type,
                "simulation_type": args.simulation_type,
                "infer_run_name":run_name_string,
                "infer_run_num": args.infer_run
            }
        else:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies,
                "config_mode": args.config_mode,
                "datetime": args.datetime,
                "inference_set": 'imagesTs',
                #Params corresponding to the inference_setup.py script.
                "pre_transforms_version_param": args.pre_transforms_version_param,
                "inferer_version_param": args.inferer_version_param,
                "inverse_transforms_version_param": args.inverse_transforms_version_param,
                "post_transforms_version_param": args.post_transforms_version_param,
                #Params corresponding to the test click simulation for inference.
                "simulation_transf_config_parametrisation": args.simulation_transf_config_parametrisation,
                "infer_click_parametrised_bool": args.infer_click_parametrised_bool,
                "infer_click_parametrisations": args.infer_click_parametrisations,
                "infer_click_parametrisations_string": inference_click_parametrisation_string,
                #Introducing the vrsion params for the config files
                "network_version_param": args.network_version_param,
                "infer_config_version_param": args.infer_config_version_param,
                "strategy_method_version_param": args.strategy_method_version_param,
                "scoring_method_version_param": args.scoring_method_version_param,
                #Introducing name of the actual validation/test + the simulation type probablistic or deterministic click generation.
                "infer_type": args.infer_type,
                "simulation_type": args.simulation_type,
                "infer_run_name":run_name_string,
                "infer_run_num": args.infer_run

            }


    upper_level_dataset_dir = os.path.join(base_directory, 'datasets', args.studies) #



    #Test set inference
    if args.infer_type == "test":
        
        ############# Creating the string for the model version so that we can save each output segmentation problem into its own separate folder ############
        model_version = args.datetime 
        model_checkpoint = args.checkpoint if args.checkpoint != None else 'best_val_score_epoch'
        
        original_test_set_dir = os.path.join(upper_level_dataset_dir, 'imagesTs')
        new_test_set_dir = os.path.join(upper_level_dataset_dir, 'test' + f'_{args.simulation_type}', args.datetime, model_checkpoint, run_name_string, inference_click_parametrisation_string, f'run_{args.infer_run}')
        #Create a separate folder copy within our results folder (more compatible with the current datastore object and methods)
        
        if os.path.exists(new_test_set_dir):
            shutil.rmtree(new_test_set_dir)
        shutil.copytree(original_test_set_dir, new_test_set_dir)

        app = MyApp(app_dir, new_test_set_dir, conf)

        '''
        Autoseg request format: Infer Request: {'model': 'deepeditplusplus_seg', 'image': 'spleen_32', 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Deepedit request format: Infer Request: {'model': 'deepeditplusplus', 'image': 'spleen_32', 'background': [[279, 255, 66]], 'spleen': [], 'label': 'spleen', 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300, 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Interactive Init request format: Infer Request: {'model': 'deepeditplusplus_interactive_init', 'image': 'spleen_32', 'label':'background': [], 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        
        '''
        
        ################# REQUEST TEMPLATES: ####################

        ### MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 .
        # This may potentially only be implemented once it is on the UI
        #This may enable the code to retain some pre-transforms which aren't randomizable in memory (e.g. normalisation/etc.)

        request_templates = dict()

        request_templates['Autoseg_template'] = {'model': args.model + '_autoseg', 'result_dtype': 'uint8','client_id': 'user-xyz', "restore_label_idx": False}
        request_templates['Interactive_template'] = {'model': args.model + '_interactive_init', 'result_dtype': 'uint8', 'client_id': 'user-xyz', "restore_label_idx": False}
        request_templates['Editing_template'] = {'model': args.model, 'result_dtype': 'uint8', 'client_id': 'user-xyz', "restore_label_idx": False}


        ############### Loading the label configuration. This extracts the class-labels / integer codes dictionary. #########
        label_config_path = os.path.join(upper_level_dataset_dir, 'label_configs.txt')
        with open(label_config_path) as f:
            label_configs = json.load(f)

        #####################################################
        
        # Run on all devices
        for device in device_list():

            # while True:
            #     sample = app.next_sample(request={"strategy": "first"})
            #     if sample == {}:
            #         break
            #     image_id = sample["id"]
            #     image_path = sample["path"] 
                
            #     inner_loop_runner(app, request_templates= request_templates, task_configs= task, image_info = [image_id, image_path], device = device, output_dir = studies_test_dir, label_configs = label_configs)

            #break

            #For every image in the new test set dir that is named with a .nii.gz ending, we perform the segmentations! 

            #Image names ending with .nii.gz 
            
            image_names = [x for x in os.listdir(new_test_set_dir) if x.endswith('.nii.gz')]
            image_names.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string))))
            for image_name in image_names:
                
                #Extracting the image name without the file extension:
                image_id = image_name.split('.')[0] 
                image_path = os.path.join(new_test_set_dir, image_name)

                #Here we create a dict containing the information about the inference configurations
                inference_config = dict() 

                inference_config['Infer Run Name'] = args.infer_run_name
                inference_config['Image Info'] = [image_id, image_path]
                inference_config['Output Dir'] = new_test_set_dir
                inference_config['Class Label Config'] = label_configs
                inference_config['Sequentiality Mode'] = args.sequentiality_mode

                #We create a click simulation config which takes the information required for producing the clicks and the sanity check gt folders.
                
                click_sim_config = run_generate_test_config_dict(args, args.simulation_transf_config_parametrisation, inference_click_parametrisations_dict)
            

                inference_config['Click Simulation Configs'] = click_sim_config

                if args.simulation_type == 'probabilistic':
                    
                    click_simulation_class = probabilistic_click_simulation(label_configs, click_sim_config)

                    probabilistic_inner_loop_runner(app, request_templates=request_templates, device=device, inference_run_configs=inference_config, click_simulation_class=click_simulation_class)
                else:
                    pass 
                    #Requires deterministic implementation! 
        return

    if args.infer_type == "validation":
        

        ############### Extracting Infer run segmentation save folders so that different runs can be executed + have labels saved separately #################

        with open(os.path.join(upper_level_dataset_dir, "train_val_split_dataset.json")) as f:
            dictionary_setting = json.load(f)
            val_dataset = dictionary_setting[f"fold_{args.val_fold}"]

        ############# Creating the string for the model version so that we can save each output segmentation problem into its own separate folder ############

        model_version = args.datetime 
        model_checkpoint = args.checkpoint if args.checkpoint != None else 'best_val_score_epoch'
        
        new_test_dir = os.path.join(upper_level_dataset_dir, 'validation' + f'_{args.simulation_type}', args.datetime, model_checkpoint, run_name_string, inference_click_parametrisation_string, f'run_{args.infer_run}')
        #Create a separate folder copy within our results folder (more compatible with the current datastore object and methods)
        
        
        if os.path.exists(new_test_dir):
            shutil.rmtree(new_test_dir)
        os.makedirs(new_test_dir)

        ########## Joining the subdir/image-labels     
        for pair_dict in val_dataset:
            
            #Extracting the paths without the extension, for the image - label pairs.
            pair_dict["image"] = os.path.join(upper_level_dataset_dir, pair_dict["image"][2:])
            pair_dict["label"] = os.path.join(upper_level_dataset_dir, pair_dict["label"][2:])


            #copy all the images over:
            image_name_path_no_ext = pair_dict["image"]
            #copy the labels over into the "original" folder.
            label_name_path_no_ext = pair_dict["label"]

            #Extracting the i.d. without the extension.
            image_id = image_name_path_no_ext.split('/')[-1]

            output_directory_path = new_test_dir
            
            #copy the images over
            shutil.copy(image_name_path_no_ext + '.nii.gz', os.path.join(output_directory_path, image_id + '.nii.gz'))

            #copy the ground truths over
            os.makedirs(os.path.join(output_directory_path, 'labels', 'original'), exist_ok=True)

            shutil.copy(label_name_path_no_ext + '.nii.gz', os.path.join(output_directory_path, 'labels', 'original', image_id + '.nii.gz'))
 
        app = MyApp(app_dir, new_test_dir, conf)

        '''
        Autoseg request format: Infer Request: {'model': 'deepeditplusplus_seg', 'image': 'spleen_32', 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Deepedit request format: Infer Request: {'model': 'deepeditplusplus', 'image': 'spleen_32', 'background': [[279, 255, 66]], 'spleen': [], 'label': 'spleen', 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300, 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Interactive Init request format: Infer Request: {'model': 'deepeditplusplus_interactive_init', 'image': 'spleen_32', 'label':'background': [], 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        
        '''
        
        ################# REQUEST TEMPLATES: ####################

        ### MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 .
        # This may potentially only be implemented once it is on the UI
        #This may enable the code to retain some pre-transforms which aren't randomizable in memory (e.g. normalisation/etc.)

        request_templates = dict()

        request_templates['Autoseg_template'] = {'model': args.model + '_autoseg', 'result_dtype': 'uint8','client_id': 'user-xyz', "restore_label_idx": False}
        request_templates['Interactive_template'] = {'model': args.model + '_interactive_init', 'result_dtype': 'uint8', 'client_id': 'user-xyz', "restore_label_idx": False}
        request_templates['Editing_template'] = {'model': args.model, 'result_dtype': 'uint8', 'client_id': 'user-xyz', "restore_label_idx": False}


        ############### Loading the label configuration. This extracts the class-labels / integer codes dictionary. #########
        label_config_path = os.path.join(upper_level_dataset_dir, 'label_configs.txt')
        with open(label_config_path) as f:
            label_configs = json.load(f)['labels']

        #####################################################
        
        # Run on all devices
        for device in device_list():

            # while True:
            #     sample = app.next_sample(request={"strategy": "first"})
            #     if sample == {}:
            #         break
            #     image_id = sample["id"]
            #     image_path = sample["path"] 
                
            #     inner_loop_runner(app, request_templates= request_templates, task_configs= task, image_info = [image_id, image_path], device = device, output_dir = studies_test_dir, label_configs = label_configs)

            #break

            #For every image in the new test set dir that is named with a .nii.gz ending, we perform the segmentations! 

            #Image names ending with .nii.gz 
            
            image_names = [x for x in os.listdir(new_test_dir) if x.endswith('.nii.gz')]
            image_names.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string))))

            for image_name in image_names:
                
                #Extracting the image name without the file extension:
                image_id = image_name.split('.')[0] 
                image_path = os.path.join(new_test_dir, image_name)

                #Here we create a dict containing the information about the inference configurations
                inference_config = dict() 

                inference_config['Infer Run Name'] = args.infer_run_name
                inference_config['Image Info'] = [image_id, image_path]
                inference_config['Output Dir'] = new_test_dir
                inference_config['Class Label Config'] = label_configs
                inference_config['Sequentiality Mode'] = args.sequentiality_mode

                #We create a click simulation config which takes the information required for producing the clicks and the sanity check gt folders.
                
                click_sim_config = run_generate_test_config_dict(args, args.simulation_transf_config_parametrisation, inference_click_parametrisations_dict)
            

                inference_config['Click Simulation Configs'] = click_sim_config

                if args.simulation_type == 'probabilistic':
                    
                    click_simulation_class = probabilistic_click_simulation(label_configs, click_sim_config)

                    probabilistic_inner_loop_runner(app, request_templates=request_templates, device=device, inference_run_configs=inference_config, click_simulation_class=click_simulation_class)
                else:
                    pass 
                    #Requires deterministic implementation!
        
        return
    

if __name__ == "__main__":

    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()
