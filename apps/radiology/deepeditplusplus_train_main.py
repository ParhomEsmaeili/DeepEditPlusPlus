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
from typing import Dict
import datetime 
############################
from os.path import dirname as up
import os
import sys
file_dir = up(up(up(os.path.abspath(__file__))))
sys.path.append(file_dir)
print(sys.path)
import torch 
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
# from monailabel.utils.others.planner import HeuristicPlanner

logger = logging.getLogger(__name__)

########### Additional modules for the inference and training scripts ############
from monailabel.utils.others.generic import device_list, file_ext
import ast

########## Imports for heuristicplanner checker

deepeditpp_runscript_utils_dir = os.path.join(up(os.path.abspath(__file__)), 'lib', 'app_utils', 'deepeditplusplus_run_script_utils')
sys.path.append(deepeditpp_runscript_utils_dir)
from deepeditplusplus_heuristic_planner_utils.planner_train import HeuristicPlanner
from deepeditplusplus_heuristic_planner_utils.checking_heuristic_planner import HeuristicPlannerChecker

######## Imports for general utilities required

from general_utils.extract_train_val_lists import load_data_split_lists

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf, planner):
        self.model_dir = os.path.join(app_dir, "model")
        self.planner = planner
        print('app dir {}'.format(app_dir))
        print('studies {}'.format(studies))
        print('conf {}'.format(conf))
        configs = {}
        
        for c in get_class_names(lib.configs, "TaskConfig"):
        # for c in get_class_names()
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
        # if self.heuristic_planner:
        #     self.planner.run(datastore)
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}

        logger.info(infers)
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t


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

    codebase_directory = up(up(up(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()

    #Arguments which configure some of the actual parameters used for training and dataset. 
    parser.add_argument("--studies", default = "BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised") 
    parser.add_argument("--datetime", default = None)
    parser.add_argument("--heuristic_planner", default="False")
    parser.add_argument("--heuristic_planner_version", default='1')
    parser.add_argument("--model", default="deepeditplusplus")
    parser.add_argument("--config_mode", default="train")# choices=("train"))
    parser.add_argument("--max_epoch", default="300")
    parser.add_argument("--train_batch_size", default='1')
    parser.add_argument("--val_batch_size", default='1')
    parser.add_argument("--save_interval", default='5')
    parser.add_argument("--target_spacing", default='[1,1,1]')
    parser.add_argument("--spatial_size", default='[128,128,128]')
    parser.add_argument("--divisible_padding_factor", default='[64,64,32]')
    parser.add_argument("--val_fold", default='0', help="The fold which is designated as the validation, everything else is the train split, second value denotes how many total folds")
    parser.add_argument("--train_folds", nargs='+', default=['1','2','3','4'])
    parser.add_argument("--max_iterations", default='1')
    parser.add_argument("--interactive_init_prob_train", default='1/2')
    parser.add_argument("--deepedit_prob_train", default="1/3")
    parser.add_argument("--interactive_init_prob_val", default="0")
    parser.add_argument("--deepedit_prob_val", default="1")
    
    
    #Introducing the version params for setting up training_setup.py, the config_setup.py version param, and the network selected version_param.
    
    parser.add_argument("--optimizer_version_param", default='0')
    parser.add_argument("--lr_scheduler_version_param", default='0')
    parser.add_argument("--loss_func_version_param", default='-1')
    parser.add_argument("--get_click_version_param", default='2')
    parser.add_argument("--train_pre_transforms_version_param", default='-7')
    parser.add_argument("--train_post_transforms_version_param", default='2')
    parser.add_argument("--val_pre_transforms_version_param", default='-3')
    parser.add_argument("--val_post_transforms_version_param", default='1')
    parser.add_argument("--train_inferer_version_param", default='0')
    parser.add_argument("--val_inferer_version_param", default='0')
    parser.add_argument("--train_iter_update_version_param", default='-1')
    parser.add_argument("--val_iter_update_version_param", default='-1')
    parser.add_argument("--train_key_metric_version_param", default='1')
    parser.add_argument("--val_key_metric_version_param", default='1')
    parser.add_argument("--train_handler_version_param", default='0')
    parser.add_argument("--engine_version_param", default='0')

    parser.add_argument("--train_config_version_param", default='1')
    parser.add_argument("--network_version_param", default='-5') 
    parser.add_argument("--strategy_method_version_param", default='0')
    parser.add_argument("--scoring_method_version_param", default='0')


        

    args = parser.parse_args()

    app_dir = up(__file__)

    dataset_dir_tr = os.path.join(codebase_directory,'datasets', args.studies, 'imagesTr')
    
    ################ Adding the name of the cuda device used ###############
    cuda_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(cuda_device)
    # print(device_name)


    ############### Extracting the datetime being used for model training (if its a completely fresh one then it would be None by default)

    if args.datetime == None:
        input_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        use_pretrained_model = "False"

    else: 
        input_datetime = args.datetime
        use_pretrained_model = "True"

############################################# Heuristic Planner Operations  ############################################################################

    #Checking that all of the necessary checks have been made with respect to the heuristic planner (e.g. whether it is needed or not, etc.):

    HeuristicPlannerChecker(vars(args))()

    # Use Heuristic Planner to extract information regarding the dataset, gpu capabilities etc for fingerprinting etc.....

   if strtobool(args.heuristic_planner):
        planner = HeuristicPlanner(version_param=args.heuristic_planner_version)

        planner.run(vars(args))
        planner_dict = vars(args)

    else:
        planner = None 
        planner_dict = dict()
    ############### Adding flexibility for selecting which set of model weights to use #####################
    

    conf = {
        "models": args.model,
        "use_pretrained_model": use_pretrained_model,
        "datetime":input_datetime,
        "heuristic_planner_bool":args.heuristic_planner,
        "heuristic_planner_version": args.heuristic_planner_version,
        "heuristic_planner_dict": planner_dict,

        "dataset_name": args.studies,
        "max_epochs": args.max_epoch,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "save_interval": args.save_interval,
        "config_mode": args.config_mode,
        "target_spacing":args.target_spacing,
        "spatial_size":args.spatial_size,
        "divisible_padding_factor":args.divisible_padding_factor,
        "val_fold":args.val_fold,
        "train_folds":args.train_folds,
        "max_iterations":args.max_iterations,
        "interactive_init_prob_train":args.interactive_init_prob_train,
        "deepedit_prob_train":args.deepedit_prob_train,
        "interactive_init_prob_val":args.interactive_init_prob_val,
        "deepedit_prob_val":args.deepedit_prob_val,
        
        #Introducing the version params for setting up the components in the config file.
    
        "optimizer_version_param": args.optimizer_version_param,
        "lr_scheduler_version_param": args.lr_scheduler_version_param,
        "loss_func_version_param": args.loss_func_version_param,
        "get_click_version_param": args.get_click_version_param,
        "train_pre_transforms_version_param": args.train_pre_transforms_version_param,
        "train_post_transforms_version_param": args.train_post_transforms_version_param,
        "val_pre_transforms_version_param": args.val_pre_transforms_version_param,
        "val_post_transforms_version_param": args.val_post_transforms_version_param,
        "train_inferer_version_param": args.train_inferer_version_param,
        "val_inferer_version_param": args.val_inferer_version_param,
        "train_iter_update_version_param": args.train_iter_update_version_param,
        "val_iter_update_version_param": args.val_iter_update_version_param,
        "train_key_metric_version_param": args.train_key_metric_version_param,
        "val_key_metric_version_param": args.val_key_metric_version_param,
        "train_handler_version_param": args.train_handler_version_param,
        "engine_version_param": args.engine_version_param,
        "network_version_param": args.network_version_param,

        #Introducing the vrsion params for initialisation of the config components themselves.

        "train_config_version_param": args.train_config_version_param,
        "strategy_method_version_param": args.strategy_method_version_param,
        "scoring_method_version_param": args.scoring_method_version_param
    }

    # Train
    app = MyApp(app_dir, dataset_dir_tr, conf, planner)

    # ########## Loading in the list of train/val images #######################

    # val_fold = args.val_fold
    # train_folds = args.train_folds
    
    # dataset_dir_outer = os.path.join(codebase_directory, "datasets", args.studies) 
    # #The dataset folder which contains alllll of the information, and not just the imagesTr folder.

    # with open(os.path.join(dataset_dir_outer, "train_val_split_dataset.json")) as f:
    #     dictionary_setting = json.load(f)
    #     val_dataset = dictionary_setting[f"fold_{val_fold}"]
    #     training_dataset = []
    #     for i in train_folds:
    #         # if i != int(val_fold):
    #         training_dataset += dictionary_setting[f"fold_{i}"]

    # ########## Joining the subdir/image-labels     
    # for pair_dict in val_dataset:
    #     pair_dict["image"] = os.path.join(dataset_dir_outer, pair_dict["image"][2:]) + '.nii.gz'
    #     pair_dict["label"] = os.path.join(dataset_dir_outer, pair_dict["label"][2:]) + '.nii.gz'

    # for pair_dict in training_dataset:
    #     pair_dict["image"] = os.path.join(dataset_dir_outer, pair_dict["image"][2:]) + '.nii.gz'
    #     pair_dict["label"] = os.path.join(dataset_dir_outer, pair_dict["label"][2:]) + '.nii.gz'

    ############## Extracting the train and validation fold datalists #######################################

    training_dataset, val_dataset = load_data_split_lists(args.train_folds, args.val_fold, args.studies) #Last one = dataset folder name.


    # if strtobool(args.heuristic_planner):
    #     planner_dict = vars(app.planner)
    # else:
    #     planner_dict = dict()

    app.train(
        request={
            "model": args.model,
            "max_epochs": int(args.max_epoch),
            "dataset": "SmartCacheDataset", #"Dataset",  # PersistentDataset, CacheDataset
            "early_stop_patience":-1,
            "train_batch_size": int(args.train_batch_size),
            "val_batch_size": int(args.val_batch_size),
            "multi_gpu": False,
            "gpus":"all",
            "dataloader":"ThreadDataLoader",
            "tracking":"mlflow",
            "tracking_uri":"",
            "tracking_experiment_name":"",
            "client_id":"user-xyz",
            "name":"train_01",
            "pretrained" : False,
            "device": device_name, #
            "local_rank": 0,
            "train_ds":training_dataset,
            "val_ds":val_dataset,
            "planner_dict": planner_dict
        },
    )


if __name__ == "__main__":

    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()
