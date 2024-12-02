import logging
import os
from typing import Any, Dict, Optional, Union

# from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
# from monailabel.interfaces.tasks.strategy import Strategy
# from monailabel.interfaces.tasks.train import TrainTask
# from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
# from monailabel.utils.others.generic import download_file, strtobool

def run_scoring_method(version_param, networks_dict, infer_class, self_dict):
    
    supported_versions = ['0']

    assert version_param in supported_versions 

    if version_param == '0':

        methods: Dict[str, ScoringMethod] = {
                    "dice": Dice(),
                    "sum": Sum(),
                }

        if self_dict['epistemic_enabled']:
            methods[f"{self_dict['name']}_epistemic"] = EpistemicScoring(
                model=self_dict['path'],
                network=networks_dict['epistemic_network'],
                transforms=infer_class(
                    type=InferType.SEGMENTATION,
                    path=self_dict['path'],
                    network=self_dict['infer_network'],
                    labels=self_dict['labels'],
                    preload=strtobool(self_dict['conf'].get("preload", "false")),
                    spatial_size=self_dict['spatial_size'],
                    imaging_modality=self_dict['imaging_modality']
                ).pre_transforms(),
                num_samples=self_dict['epistemic_samples'],
            )
        return methods