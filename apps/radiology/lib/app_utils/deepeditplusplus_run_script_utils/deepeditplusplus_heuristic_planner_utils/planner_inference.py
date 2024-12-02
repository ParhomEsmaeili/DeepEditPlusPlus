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

import logging
import random

import numpy as np
from monai.transforms import LoadImage
from tqdm import tqdm

from monailabel.utils.others.generic import gpu_memory_map

logger = logging.getLogger(__name__)


class HeuristicPlanner:
    def __init__(self, version_param:str ='1'):

        supported_version_params = ['1']
        if version_param not in supported_version_params:
            raise ValueError('Not a supported Heuristic Planner')

        self.target_spacing = None #target_spacing
        self.spatial_size = None # spatial_size
        self.max_pix = None
        self.min_pix = None
        self.lower_bound = None
        self.upper_bound = None
        self.mean_pix = None
        self.std_pix = None


        self.version_param = version_param

    def run(self, datastore):
        
        if self.version_param == '1':
            raise NotImplementedError 
