import logging
import os
from typing import Any, Dict, Optional, Union

# from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
# from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
# from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
# from monailabel.tasks.scoring.dice import Dice
# from monailabel.tasks.scoring.epistemic import EpistemicScoring
# from monailabel.tasks.scoring.sum import Sum
# from monailabel.utils.others.generic import download_file, strtobool

def run_strategy_method(version_param, self_dict):
    
    supported_versions = ['0']

    assert version_param in supported_versions 

    if version_param == '0':

        strategies: Dict[str, Strategy] = {}
        if self_dict['epistemic_enabled']:
            strategies[f"{self_dict['name']}_epistemic"] = Epistemic()
        return strategies