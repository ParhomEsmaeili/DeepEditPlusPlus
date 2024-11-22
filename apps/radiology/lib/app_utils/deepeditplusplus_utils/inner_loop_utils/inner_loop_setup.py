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

from __future__ import annotations

from os.path import dirname as up
import os
import sys
file_dir = up(os.path.abspath(__file__))
sys.path.append(file_dir)
engines_dir = os.path.join(up(up(up(up(up(up(up(os.path.abspath(__file__)))))))), 'engines')
sys.path.append(engines_dir)

from inner_loop_utils.inner_loop_version_minus_1 import run as inner_loop_minus_1_run
from inner_loop_utils.inner_loop_version_1 import run as inner_loop_1_run
from inner_loop_utils.inner_loop_version_2 import run as inner_loop_2_run
from inner_loop_utils.inner_loop_version_3 import run as inner_loop_3_run
from inner_loop_utils.inner_loop_version_4 import run as inner_loop_4_run

########################

from collections.abc import Callable, Sequence

import numpy as np
import torch
import logging 
from monai.data import decollate_batch, list_data_collate
# from monai.engines import SupervisedEvaluator, SupervisedTrainer
from engines.standard_engines.trainer import SupervisedTrainer as DefaultSupervisedTrainer
from engines.standard_engines.evaluator import SupervisedEvaluator as DefaultSupervisedEvaluator 

# from monai.engines import SupervisedTrainer as DefaultSupervisedTrainer
# from monai.engines import SupervisedEvaluator as DefaultSupervisedEvaluator

#Importing the interactive version..
from engines.interactive_seg_engines.trainer import SupervisedTrainer as InteractiveSupervisedTrainer
from engines.interactive_seg_engines.evaluator import SupervisedEvaluator as InteractiveSupervisedEvaluator 

from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys


'''
Version param denotes the version of this interaction/inner loop functionality that is being used for training. 

Version -1: A fully autoseg implementation (i.e. it does nothing.)
Version 0: DeepEdit original implementation. (not yet re-implemented)
Version 1: DeepEdit++ v1.1 version.
Version 2: Approximate loop unrolling (randomly selected the max-iter value for each training iteration from the max-iter param as an upper limit.)
Version 3: Approximate loop unrolling with click sets generated and passed into the engine.iteration function for performing multi-headed click-based losses.
Version 4: Full loop unrolling with click sets generated and passed into the engine.iteration function for performing multi-headed click-based losses.
'''

class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for DeepEditlike Training/Evaluation.

    Args:

        num_intensity_channel: The number of channels in the input image itself (hard-fixed to 1 for DeepEdit++ currently)
        
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        
        click_probability_key: key to click/interaction probability

        train: True for training mode or False for evaluation mode

        external_validation_output_dir: the directory in which the validations are saved in a hacky method (to generate validation curves across each use mode)

        version_param: The version of the inner loop that is being used.

        self_dict, contains all of the variables from the parent script self variable. 
            This includes the dict which contains all of the component parametrisations:
    
            interactive_init_probability: probability of simulating clicks in an iteration for the initialisation (or for the entirety inner loop in the original DeepEdit)
            deepedit_probability: probability of simulating the editing clicks for the inner loop (DeepEdit++)
         
            ^ provided for both " "_train and " "_val 

            max_iterations: maximum number of interactive editing iterations per training iteration/val iteration sample

    """

    def __init__(
        self,
        # interactive_init_probability: float,
        # deepedit_probability: float,
        self_dict,
        num_intensity_channel: int,
        transforms: Sequence[Callable] | Callable | None,
        train: bool,
        #label_names: None | dict[str, int] = None,
        click_probability_key: str | None = None,
        # max_iterations: int = 1,
        external_validation_output_dir:str | None = None,
        version_param: str = '1'
        ) -> None:


        self.num_intensity_channel = num_intensity_channel
        self.transforms = Compose(transforms) if not isinstance(transforms, Compose) else transforms
        self.train = train
        #self.label_names = label_names
        self.click_probability_key = click_probability_key
        self.external_validation_output_dir = external_validation_output_dir
        self.version_param = version_param 

        # self.required_inner_loop_parametrisations = ['max_iterations', 'deepedit_probability', 'interactive_init_probability']

        self.max_iterations = self_dict['component_parametrisation_dict']['max_iterations'] #We assume any variation between train and val will just be implemented in the 
        #inner loop version code with regards to the number of edit iterations being performed for validation.

        if self.train:
            #In this case, we set the values for each parametrisation according to the train parametrisations
            self.interactive_init_probability = self_dict['component_parametrisation_dict']['interactive_init_probability_train']
            self.deepedit_probability = self_dict['component_parametrisation_dict']['deepedit_probability_train']

        else:
            #In this case, we set the values for each parametrisation according to the train parametrisations
            self.interactive_init_probability = self_dict['component_parametrisation_dict']['interactive_init_probability_val']
            self.deepedit_probability = self_dict['component_parametrisation_dict']['deepedit_probability_val']

        self.supported_inner_loop_versions = ['-1', '1','2', '3', '4']

        # self.self_dict = dict(vars(self)) #Creating a dictionary from the self attributes, this will be fed forward into the 

        assert self.version_param in self.supported_inner_loop_versions
    

    def __call__(self, 
                engine: DefaultSupervisedTrainer | DefaultSupervisedEvaluator | InteractiveSupervisedTrainer | InteractiveSupervisedEvaluator, 
                batchdata: dict[str, torch.Tensor]) -> dict:
        
        if self.version_param == '-1':

            return inner_loop_minus_1_run(dict(vars(self)), engine, batchdata)

        elif self.version_param == '1':

            #The train/val mode should be partitioned in the actual train setup.py script. The same params are being used but the value assigned will differ! 
            
            return inner_loop_1_run(dict(vars(self)), engine, batchdata)

        elif self.version_param == '2':

            return inner_loop_2_run(dict(vars(self)), engine, batchdata)

        elif self.version_param == '3':

            return inner_loop_3_run(dict(vars(self)), engine, batchdata)

        elif self.version_param == '4':

            return inner_loop_4_run(dict(vars(self)), engine, batchdata)

            
            
