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
########################

import lib.configs
from lib.activelearning import Last
from lib.infers.deepgrow_pipeline import InferDeepgrowPipeline
from lib.infers.vertebra_pipeline import InferVertebraPipeline


import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.scribbles.infer import GMMBasedGraphCut, HistogramBasedGraphCut
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
import copy


########## Additional modules and packages for the inference click simulation
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ToNumpyd,
    Resized
) 
from monailabel.deepeditPlus.retooled_transforms import (
    NormalizeLabelsInDatasetd,
    MappingLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    FindDiscrepancyRegionsDeepEditd,
    AddRandomGuidanceDeepEditd,
    AddRandomGuidanceDeepEditdFixed
)

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
        if self.bundles:
            for n, b in self.bundles.items():
                i = BundleInferTask(b, self.conf)
                logger.info(f"+++ Adding Bundle Inferer:: {n} => {i}")
                infers[n] = i

        #################################################
        # Scribbles
        #################################################
        if self.scribbles:
            infers.update(
                {
                    "Histogram+GraphCut": HistogramBasedGraphCut(
                        intensity_range=(-300, 200, 0.0, 1.0, True),
                        pix_dim=(2.5, 2.5, 5.0),
                        lamda=1.0,
                        sigma=0.1,
                        num_bins=64,
                        labels=task_config.labels,
                    ),
                    "GMM+GraphCut": GMMBasedGraphCut(
                        intensity_range=(-300, 200, 0.0, 1.0, True),
                        pix_dim=(2.5, 2.5, 5.0),
                        lamda=5.0,
                        sigma=0.5,
                        num_mixtures=20,
                        labels=task_config.labels,
                    ),
                }
            )

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

        #################################################
        # Bundle Models
        #################################################
        if self.bundles:
            for n, b in self.bundles.items():
                t = BundleTrainTask(b, self.conf)
                if not t or not t.is_valid():
                    continue

                logger.info(f"+++ Adding Bundle Trainer:: {n} => {t}")
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


    #####################################################################
def inner_loop_runner(app, request_templates, task_configs, image_info, device, output_dir, label_configs):
    '''
    Args:
    
    request_templates: the basic templates for the inference requests of the two deepedit modes
    task_configs: List of the task specific configurations (1) initial_task mode and if appropriate, 2) subsequent mode + 3) no.iters.)
    image_info: List of the image information, (1) image id, (2) image_path to save to(?)
    device: device that is being used to perform inference

    Only TWO actual modes: Autoseg/Interactive Seg-Deepediting. But we will treat the initial deepedit as deepgrow (or conversely, deepedit = extended deepgrow)
    '''

    image_id = image_info[0]
    image_path = image_info[1]

    if len(task_configs) > 1:
        initial_task = task_configs[1]
        subsequent_task = task_configs[0]
        num_iterations = int(task_configs[2])

        #Extracting the appropriate request formats:
        initial_request = request_templates[initial_task + "_template"]
        subsequent_request = request_templates[subsequent_task + "_template"]
        
        #Appending the key:val pair for the image id.
        initial_request["image"] = image_id
        subsequent_request["image"] = image_id 

        #Appending the device name:
        initial_request["device"] = device
        subsequent_request["device"] = device

        #Extracting the path for the GT label that we need to simulate the clicks with (if necessary):
        gt_path = os.path.join(output_dir, 'labels', 'original', image_id + '.nii.gz')

        #Creating a save folder for the RAS and resized label to verify the click simulation is done appropriately
        save_folder = os.path.join(up(up(gt_path)), 'RAS_GT')
        os.makedirs(save_folder, exist_ok=True)

        #Creating a save folder for storing the guidance points in order to debug the performance of the models.
        
        guidance_points_save_folder = os.path.join(up(up(gt_path)), f'guidance_points')
        os.makedirs(guidance_points_save_folder, exist_ok=True) 


        if initial_task == "autoseg":
            res = app.infer(request = initial_request)  #request={"model": args.model, "image": image_id, "device": device})
            tracked_guidance = dict()
            for key in label_configs["labels"].keys(): 
                tracked_guidance[key] = []
            

        elif initial_task == "deepgrow":
            initial_request, transform_output_dict, tracked_guidance = click_simulation(initial_request, label_configs, clicking_task = initial_task, gt_path=gt_path, tracked_guidance=None)
            res = app.infer(request = initial_request)
        # Have a saved label for the initial seg, so that we have a measure for how the iterations evolve with accuracy
            
            
            #Saving the guidance points (RAS orientation) for debugging the model performance/sanity checking. 
            guidance_points_save = dict()
            for label_name in label_configs["labels"].keys():
                guidance_points_save[label_name] = initial_request[label_name]
            
            # saved_dict = dict()
            # saved_dict[image_id] = guidance_points_save

            try:
                with open(os.path.join(guidance_points_save_folder, initial_task + '.json'), 'r') as f:
                    saved_dict = json.load(f)
                    saved_dict[image_id] = guidance_points_save
                with open(os.path.join(guidance_points_save_folder, initial_task + '.json'), 'w') as f:
                    json.dump(saved_dict, f)
            except:
                with open(os.path.join(guidance_points_save_folder, initial_task + '.json'), 'w') as f:
                    # saved_dict = json.load(f)
                    saved_dict = dict()
                    saved_dict[image_id] = guidance_points_save 
                    json.dump(saved_dict, f)


        res_savepath = label_saving(res, os.path.join(output_dir, "labels", initial_task), image_id, image_path)
        



        for i in range(1, num_iterations + 1):
            
            subsequent_request["previous_seg"] = res_savepath #res["file"]
            tracked_guidance_input = copy.deepcopy(tracked_guidance)
            # The function which will extract click points given the current instance of the label. 
            subsequent_request, transform_output_dict, discrepancy_output_dict, tracked_guidance = click_simulation(subsequent_request, label_configs, clicking_task = subsequent_task, gt_path = gt_path, tracked_guidance=tracked_guidance_input)
            
            #Deleting the "previous seg" key:val pair from the request: It is not needed beyond this point.
            del subsequent_request["previous_seg"]


            #Saving the guidance points (RAS orientation) for debugging the model performance/sanity checking. 
            guidance_points_save = dict()
            for label_name in label_configs["labels"].keys():
                guidance_points_save[label_name] = subsequent_request[label_name]
            
            #Saving the guidance points to a json file. 
            # saved_dict = dict()
            # saved_dict[image_id] = guidance_points_save

            if i != num_iterations:
                try:
                    with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'r') as f:
                        saved_dict = json.load(f)
                        saved_dict[image_id] = guidance_points_save
                    with open(os.path.join(guidance_points_save_folder, f'deepedit_iteration_{i}.json'), 'w') as f:
                        json.dump(saved_dict, f) 
                except:
                    with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'w') as f:
                        # saved_dict = json.load(f)
                        saved_dict = dict()
                        saved_dict[image_id] = guidance_points_save
                        json.dump(saved_dict, f) 
            else:
                try:
                    with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'r') as f:
                        saved_dict = json.load(f)
                        saved_dict[image_id] = guidance_points_save
                    with open(os.path.join(guidance_points_save_folder, f'final_iteration.json'), 'w') as f:
                        json.dump(saved_dict, f) 
                except:
                    with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'w') as f:
                        # saved_dict = json.load(f)
                        saved_dict = dict()
                        saved_dict[image_id] = guidance_points_save
                        json.dump(saved_dict, f) 




            ############### Saving the GT labels in RAS orientation for eye inspection #######################
            # save_folder = os.path.join(up(up(gt_path)), 'RAS_GT')
            # os.makedirs(save_folder, exist_ok=True)

            nib.save(nib.Nifti1Image(np.array(transform_output_dict["label"][0]), None), os.path.join(save_folder, f'{subsequent_request["image"]}' + '.nii.gz'))

            ################## Saving the discrepancies at iteration i of the editing ###########################
            #Discrepancy save folder:

            if i != num_iterations:
                discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' padded_discrepancy', 'iteration_' + str(i), image_id)
                os.makedirs(discrepancy_folder_path, exist_ok=True)

                for label_class in discrepancy_output_dict["label_names"].keys():
                    
                    nib.save(nib.Nifti1Image(np.array(discrepancy_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
            else:
                discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' padded_discrepancy', 'final', image_id)
                os.makedirs(discrepancy_folder_path, exist_ok=True)

                for label_class in discrepancy_output_dict["label_names"].keys():
                    
                    nib.save(nib.Nifti1Image(np.array(discrepancy_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
            
            #Saving the full sized discrepancy images in order to examine the validity of the guidance points generated.
            
            if i != num_iterations:
                discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' original_discrepancy', 'iteration_' + str(i), image_id)
                os.makedirs(discrepancy_folder_path, exist_ok=True)

                for label_class in transform_output_dict["label_names"].keys():
                    
                    nib.save(nib.Nifti1Image(np.array(transform_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
            else:
                discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' original_discrepancy', 'final', image_id)
                os.makedirs(discrepancy_folder_path, exist_ok=True)

                for label_class in transform_output_dict["label_names"].keys():
                    
                    nib.save(nib.Nifti1Image(np.array(transform_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
            ######################################################################################################

            # Running DeepEdit inference with the prev_label and the update clicks
            res = app.infer(request = subsequent_request)
            #For each iteration should we also have a saved label
            if i != num_iterations:
                res_savepath = label_saving(res, os.path.join(output_dir, "labels", subsequent_task + '_iteration_' + str(i)), image_id, image_path)
            else:
                _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)
        
    else:
        input_request = request_templates[task_configs[0] + "_template"]

        #Adding the image id to the input request
        input_request["image"] = image_id
        
        #Task name:
        task = task_configs[0]
        
        if task == 'autoseg':

            res = app.infer(request=input_request)
            
            _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)
            
            tracked_guidance = dict()
            for key in label_configs["labels"].keys(): 
                tracked_guidance[key] = []

        elif task == "deepgrow":
            #Extracting the path for the GT label that we need to simulate the clicks with:
            gt_path = os.path.join(output_dir, 'labels', 'original', image_id + '.nii.gz')
            
            input_request, transform_output_dict, tracked_guidance = click_simulation(input_request, label_configs, clicking_task = task, gt_path = gt_path, tracked_guidance=None)

            res = app.infer(request=input_request)
            
            _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)

            #Creating a save folder for storing the guidance points in order to debug the performance of the models.
        
            guidance_points_save_folder = os.path.join(up(up(gt_path)), f'guidance_points')
            os.makedirs(guidance_points_save_folder, exist_ok=True) 

            #Saving the guidance points (RAS orientation) for debugging the model performance/sanity checking. 
            guidance_points_save = dict()
            for label_name in label_configs["labels"].keys():
                guidance_points_save[label_name] = input_request[label_name]
                

            try:
                with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'r') as f:
                    saved_dict = json.load(f)
                    saved_dict[image_id] = guidance_points_save
                with open(os.path.join(guidance_points_save_folder, f'final_iteration.json'), 'w') as f:
                    json.dump(saved_dict, f) 
            except:
                with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'w') as f:
                    # saved_dict = json.load(f)
                    saved_dict = dict()
                    saved_dict[image_id] = guidance_points_save
                    json.dump(saved_dict, f) 

    logger.info(f'Inference completed for image: {image_id}')

def label_saving(inference_res, output_dir, image_id, image_name_path):
    #Saving the labels: 
    label = inference_res["file"]
    label_json = inference_res["params"]
    #test_dir = os.path.join(output_dir , "labels", "final")
    os.makedirs(output_dir, exist_ok=True)

    label_file = os.path.join(output_dir, image_id + file_ext(image_name_path))
    shutil.move(label, label_file)


    print(label_json)
    print(f"++++ Image File: {image_name_path}")
    print(f"++++ Label File: {label_file}")
    
    return label_file

def click_simulation(inference_request, label_configs, clicking_task, gt_path, tracked_guidance):
    '''
    label_configs contain the dict which contains all the information regarding the label configurations and mappings.

    inference request is the dict containing the base structure for the deepedit/deepgrow inference request, without the addition of the 
    click points/guidance points.

    guidance is the previously provided guidance (IF IT HAS ONE!)
    '''
    # labels = label_configs["labels"]
    # original_dataset_labels = label_configs["original_dataset_labels"]
    # label_mapping = label_configs["label_mapping"]

    input_dict = dict()

    input_dict["labels"] = label_configs["labels"]
    input_dict["original_dataset_labels"] = label_configs["original_dataset_labels"]
    input_dict["label_mapping"] = label_configs["label_mapping"]
    #GT label path
    input_dict["label"] = gt_path

    #I.e. if we are editing from a deepgrow initialisation
    if clicking_task == "deepedit":
        input_dict["guidance"] = tracked_guidance 

    if clicking_task == "deepgrow":
        composed_transform = [
        LoadImaged(keys=("label"), reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=("label")),
        MappingLabelsInDatasetd(keys="label", original_label_names = input_dict["original_dataset_labels"], label_names = input_dict["labels"], label_mapping=input_dict["label_mapping"]),
        NormalizeLabelsInDatasetd(keys="label", label_names=input_dict["labels"]), 
        #We must orientate to RAS so that the guidance points are in the correct coordinate system for the inference script.
        Orientationd(keys=["label"], axcodes="RAS"),
        #Resized(keys=("label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
        # Transforms for click simulation (depracated)
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids")
        ]
    elif clicking_task == "deepedit":
        #Adding the prev_seg path:
        input_dict["previous_seg"] = inference_request["previous_seg"]
        input_dict["probability"] = 1.0

        composed_transform = [
        LoadImaged(keys=("previous_seg", "label"), reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=("previous_seg", "label")),
        # Label mapping is only required for the label, not the previous seg, because (for now) the config labels of 
        # the inference output is just the current label config. Whereas the original GT label is still with the original
        # config labels. 
        MappingLabelsInDatasetd(keys=("label"), original_label_names=input_dict["original_dataset_labels"], label_names = input_dict["labels"], label_mapping= input_dict["label_mapping"]),
        # We normalise so that it is in the expected format (?)
        NormalizeLabelsInDatasetd(keys=("previous_seg", "label"), label_names= input_dict["labels"]), 
        #We need to re-orientate ourselves in RAS so that we generate guidance points in RAS coordinates:
        Orientationd(keys=("previous_seg", "label"), axcodes="RAS"),
        ToNumpyd(keys=("previous_seg", "label")),
            # Transforms for click simulation
        #TODO: verify that using pred = previous_SEG is valid.
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="previous_seg", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditdFixed(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        # DivisiblePadd(keys="label", k=[64,64,32])
        #Resized(keys=["label"], spatial_size=(128, 128, 128), mode=["nearest"]),
        ]
    
    transform_output_dict = Compose(transforms=composed_transform, map_items = False)(input_dict)
    
    ################### Above generates guidance points which are denoted under the output_dict with the guidance key ######################

    
    #Writer(label="label", json=None)(transform_output_dict)
    if clicking_task == "deepedit": 

        ########## This part is solely here to save the discrepancy maps for validation ####################
        discrepancy_resizing_dict = transform_output_dict.copy()
        #Obtaining a list of input keys for each of the classes that we want to save a discrepancy map for
        input_keys = []
        #mode_keys = []
        for label_class in discrepancy_resizing_dict["labels"].keys():
            discrepancy_resizing_dict[f"discrepancy_{label_class}"] = transform_output_dict["discrepancy"][label_class][0]
            input_keys.append(f"discrepancy_{label_class}")
            #mode_keys.append("nearest")

        discrepancy_output_dict = discrepancy_resizing_dict
        
        #splitting the original dict also, for the full sized discrepancies
        for label_class in transform_output_dict["labels"].keys():
            transform_output_dict[f"discrepancy_{label_class}"] = transform_output_dict["discrepancy"][label_class][0]
            
        #Save a final guidance separately since this original version of deepedit requires it! 
        final_guidance = dict()

        #Converting the guidance clicks to inputs for the inference script
        for key in transform_output_dict["guidance"].keys():
            sim_clicks = transform_output_dict["guidance"][key]
            sim_click_valid = []
            for point in sim_clicks:
                if len(point) == 3:
                    sim_click_valid.append(point)
                else:
                    if point[0] >=1:
                        sim_click_valid.append(point[1:])

            #sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
            inference_request[key] = sim_click_valid
            final_guidance[key] = sim_click_valid

    #    inference_request["guidance"] = transform_output_dict["guidance"]
        

        return inference_request, transform_output_dict, discrepancy_output_dict, final_guidance

    elif clicking_task == "deepgrow":
        #Save a final guidance separately since this original version of deepedit requires it!
        final_guidance = dict()

        #Converting the guidance clicks to inputs for the inference script
        for key in transform_output_dict["guidance"].keys():
            sim_clicks = ast.literal_eval(transform_output_dict["guidance"][key])
            sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
            inference_request[key] = sim_click_valid
            final_guidance[key] = sim_click_valid 
    #    inference_request["guidance"] = transform_output_dict["guidance"]
        return inference_request, transform_output_dict, final_guidance

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
    print(base_directory)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default = "datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT/imagesTr")# #"datasets/Task09_Spleen/imagesTr") #default= "datasets/Task01_BrainTumour/imagesTr") 
    parser.add_argument("-m", "--model", default="deepedit")
    parser.add_argument("-t", "--test", default="train")#"train") #"batch_infer", choices=("train", "infer", "batch_infer"))
    parser.add_argument("--infer_run", default='0')
    parser.add_argument("-ta", "--task", nargs="+", default=["deepedit", "autoseg", "3"], help="The subtask/mode which we want to execute, three modes (one is a pseudo mode: deepgrow=initial_deepedit) ")
    parser.add_argument("-e", "--max_epoch", default="250")
    parser.add_argument("-i", "--imaging_modality", default="MRI")
    parser.add_argument("--checkpoint") 
    parser.add_argument("--datetime", default='02062024_144027') #, default='27052024_123303') #THIS TOO
    parser.add_argument("--val_fold", default='0', help="The fold which is designated as the validation set, everything else is in the training set, second param = number of total folds.")
    parser.add_argument("--train_folds", nargs='+', default=['1','2','3','4'])
    
    
    args = parser.parse_args()

    app_dir = up(__file__)

    studies = os.path.join(base_directory, args.studies)
    
    ################ Adding the name of the cuda device used ###############
    cuda_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(cuda_device)
    print(device_name)


    ############### Adding flexibility for selecting which set of model weights to use #####################
    if args.test == "train":

        conf = {
            "models": args.model,
            "use_pretrained_model": "False",
            "dataset_name": args.studies[9:-9],
            "max_epochs": args.max_epoch,
            "mode": args.test,
            "val_fold":args.val_fold,
            "train_folds":args.train_folds,
        }
    elif args.test == "infer":
        if args.checkpoint != None:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies[9:-9],
                "max_epochs": args.max_epoch,
                "mode": args.test,
                "checkpoint": args.checkpoint,
                "datetime": args.datetime,
            }
        else:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies[9:-9],
                "max_epochs": args.max_epoch,
                "mode": args.test,
                "datetime": args.datetime
            }

    elif args.test == "validate":
        if args.checkpoint != None:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies[9:-9],
                "max_epochs": args.max_epoch,
                "mode": args.test,
                "checkpoint": args.checkpoint,
                "datetime": args.datetime,
            }
        else:
            conf = {
                "models": args.model,
                "use_pretrained_model": "False",
                "dataset_name": args.studies[9:-9],
                "max_epochs": args.max_epoch,
                "mode": args.test,
                "datetime": args.datetime
            }


    task = args.task

    # Infer
    if args.test == "infer":

        #If testing, and we want the studies folder to be task_specific, so we have shifted the app initialisation.
        ############### Extracting Task Name so that different tests can be executed + have labels saved separately #################

        if len(task) > 1:
            task_name = task[0] + "_" + task[1] + "_initialisation_" + "numIters_" + task[2]
        else:
            task_name = task[0]

        ############# Creating the string for the model version so that we can save each output segmentation problem into its own separate folder ############
        model_version = args.datetime 
        model_checkpoint = args.checkpoint if args.checkpoint != None else 'best_val_score_epoch'
        
        print(model_checkpoint)
        studies_test_dir = os.path.join(studies[:-9], args.model, studies[-8:] + f"_{task_name}", model_version, model_checkpoint, f'run_{args.infer_run}')
        #Create a separate folder copy for the subtask (more compatible with the current datastore object and methods)
        if os.path.exists(studies_test_dir):
            shutil.rmtree(studies_test_dir)
        shutil.copytree(studies, studies_test_dir)

        app = MyApp(app_dir, studies_test_dir, conf)

        '''
        Autoseg request format: Infer Request: {'model': 'deepedit_autoseg', 'image': 'spleen_32', 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Deepgrow/Deepedit request format: Infer Request: {'model': 'deepedit', 'image': 'spleen_32', 'background': [[279, 255, 66]], 'spleen': [], 'label': 'spleen', 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300, 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        New DeepGrow request format should be the same as Deepedit probably..
        '''
        
        ################# REQUEST TEMPLATES: ####################

        ### MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 

        request_templates = dict()

        request_templates['autoseg_template'] = {'model': args.model + '_autoseg', 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality,'client_id': 'user-xyz', "restore_label_idx": True}
        request_templates['deepgrow_template'] = {'model': args.model, 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality, 'client_id': 'user-xyz', "restore_label_idx": True}
        request_templates['deepedit_template'] = {'model': args.model, 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality, 'client_id': 'user-xyz', "restore_label_idx": True}


        #TODO: MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 
        ############################################################
        


        ############### Loading the label configuration #########
        label_config_path = os.path.join(base_directory, 'monailabel', 'deepedit', f'{args.studies[9:-9]}_label_configs.txt')
        with open(label_config_path) as f:
            label_configs = json.load(f)

        #####################################################
        
        # Run on all devices
        for device in device_list():

            while True:
                sample = app.next_sample(request={"strategy": "first"})
                if sample == {}:
                    break
                image_id = sample["id"]
                image_path = sample["path"] 
                
                inner_loop_runner(app, request_templates= request_templates, task_configs= task, image_info = [image_id, image_path], device = device, output_dir = studies_test_dir, label_configs = label_configs)

            #break
        return
    


    if args.test == "validate":

        #If testing, and we want the studies folder to be task_specific, so we have shifted the app initialisation.
        ############### Extracting Task Name so that different tests can be executed + have labels saved separately #################

        dataset_dir = os.path.join(base_directory, args.studies[:-9])
        print(dataset_dir)
        with open(os.path.join(dataset_dir, "train_val_split_dataset.json")) as f:
            dictionary_setting = json.load(f)
            val_dataset = dictionary_setting[f"fold_{args.val_fold}"]

        ########## Joining the subdir/image-labels     
        for pair_dict in val_dataset:
            pair_dict["image"] = os.path.join(dataset_dir, pair_dict["image"][2:])
            pair_dict["label"] = os.path.join(dataset_dir, pair_dict["label"][2:])





        if len(task) > 1:
            task_name = task[0] + "_" + task[1] + "_initialisation_" + "numIters_" + task[2]
        else:
            task_name = task[0]

        ############# Creating the string for the model version so that we can save each output segmentation problem into its own separate folder ############
        model_version = args.datetime 
        model_checkpoint = args.checkpoint if args.checkpoint != None else 'best_val_score_epoch'
        
        print(model_checkpoint)
        train_folds_string = '_'.join(args.train_folds)
        studies_val_dir = os.path.join(studies[:-9], 'validation', args.model, f"train_folds_{train_folds_string}_validation_fold_{args.val_fold}", f"{task_name}", model_version, model_checkpoint, f'run_{args.infer_run}')
        # #Create a separate folder copy for the subtask (more compatible with the current datastore object and methods)
        # if os.path.exists(studies_val_dir):
        #     shutil.rmtree(studies_val_dir)
        # shutil.copytree(studies, studies_val_dir)
        if os.path.exists(studies_val_dir):
            shutil.rmtree(studies_val_dir)
        os.makedirs(studies_val_dir)

        for pair_dict in val_dataset:
            #copy all the images over:
            image_name_path_no_ext = pair_dict["image"]
            #copy the labels over into the "original" folder.
            label_name_path_no_ext = pair_dict["label"]

            image_id = image_name_path_no_ext.split('/')[-1]

            output_path = studies_val_dir
            #copy the images over
            
            shutil.copy(image_name_path_no_ext + '.nii.gz', os.path.join(output_path, image_id + '.nii.gz'))

            #copy the ground truths over
            os.makedirs(os.path.join(output_path, 'labels', 'original'), exist_ok=True)
            shutil.copy(label_name_path_no_ext + '.nii.gz', os.path.join(output_path, 'labels', 'original', image_id + '.nii.gz'))
 
        app = MyApp(app_dir, studies_val_dir, conf)

        '''
        Autoseg request format: Infer Request: {'model': 'deepedit_autoseg', 'image': 'spleen_32', 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        Deepgrow/Deepedit request format: Infer Request: {'model': 'deepedit', 'image': 'spleen_32', 'background': [[279, 255, 66]], 'spleen': [], 'label': 'spleen', 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300, 'device': 'NVIDIA GeForce RTX 4090', 'result_extension': '.nrrd', 'result_dtype': 'uint8', 'client_id': 'user-xyz'}
        New DeepGrow request format should be the same as Deepedit probably..
        '''
        
        ################# REQUEST TEMPLATES: ####################

        ### MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 

        request_templates = dict()

        request_templates['autoseg_template'] = {'model': args.model + '_autoseg', 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality,'client_id': 'user-xyz', "restore_label_idx": True}
        request_templates['deepgrow_template'] = {'model': args.model, 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality, 'client_id': 'user-xyz', "restore_label_idx": True}
        request_templates['deepedit_template'] = {'model': args.model, 'result_dtype': 'uint8', 'imaging_modality':args.imaging_modality, 'client_id': 'user-xyz', "restore_label_idx": True}


        #TODO: MAY NEED TO ADD THIS: 'cache_transforms': True, 'cache_transforms_in_memory': True, 'cache_transforms_ttl': 300 
        ############################################################
        


        ############### Loading the label configuration #########
        label_config_path = os.path.join(base_directory, 'monailabel', 'deepedit', f'{args.studies[9:-9]}_label_configs.txt')
        with open(label_config_path) as f:
            label_configs = json.load(f)

        #####################################################
        
        # Run on all devices
        for device in device_list():

            while True:
                sample = app.next_sample(request={"strategy": "first"})
                if sample == {}:
                    break
                image_id = sample["id"]
                image_path = sample["path"] 

                
                inner_loop_runner(app, request_templates= request_templates, task_configs= task, image_info = [image_id, image_path], device = device, output_dir = output_path, label_configs = label_configs)

            #break
        return



    # # Batch Infer
    # if args.test == "batch_infer":

    #     app = MyApp(app_dir, studies, conf)

    #     app.batch_infer(
    #         request={
    #             "model": args.model,
    #             "multi_gpu": False,
    #             "save_label": True,
    #             "label_tag": "original",
    #             "max_workers": 1,
    #             "max_batch_size": 0,
    #         }
    #     )

    #     return

    # Train
    app = MyApp(app_dir, studies, conf)

    ########## Loading in the list of train/val images #######################
    dataset_dir = os.path.join(base_directory, args.studies[:-9])
    val_fold = args.val_fold
    train_folds = args.train_folds
    with open(os.path.join(dataset_dir, "train_val_split_dataset.json")) as f:
        dictionary_setting = json.load(f)
        val_dataset = dictionary_setting[f"fold_{val_fold}"]
        training_dataset = []
        for i in train_folds:
            # if i != int(val_fold):
            training_dataset += dictionary_setting[f"fold_{i}"]

    ########## Joining the subdir/image-labels     
    for pair_dict in val_dataset:
        pair_dict["image"] = os.path.join(dataset_dir, pair_dict["image"][2:])
        pair_dict["label"] = os.path.join(dataset_dir, pair_dict["label"][2:])

    for pair_dict in training_dataset:
        pair_dict["image"] = os.path.join(dataset_dir, pair_dict["image"][2:])
        pair_dict["label"] = os.path.join(dataset_dir, pair_dict["label"][2:])

    app.train(
        request={
            "model": args.model,
            "max_epochs": int(args.max_epoch),
            "dataset": "SmartCacheDataset", #"Dataset",  # PersistentDataset, CacheDataset
            "early_stop_patience":-1,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "multi_gpu": False,
            "gpus":"all",
            "val_split": 0.2,
            "dataloader":"ThreadDataLoader",
            "tracking":"mlflow",
            "tracking_uri":"",
            "tracking_experiment_name":"",
            "client_id":"user-xyz",
            "name":"train_01",
            "pretrained" : False,
            "device": device_name, #
            "local_rank": 0,
            "imaging_modality": args.imaging_modality,
            "train_ds":training_dataset,
            "val_ds":val_dataset
        },
    )


if __name__ == "__main__":

    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()
