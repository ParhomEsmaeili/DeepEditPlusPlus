'''
Version of the inner loop which is intended for a standard segmentation network (e.g. a basic U-Net).
'''

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import logging 
from monai.data import decollate_batch, list_data_collate
# from monai.engines import SupervisedEvaluator, SupervisedTrainer

##################################################
import nibabel as nib

from os.path import dirname as up

import os 
import sys 

file_dir = os.path.join(up(os.path.abspath(__file__)))
engines_dir = os.path.join(up(up(up(up(up(up(up(os.path.abspath(__file__)))))))), 'engines')
sys.path.append(engines_dir)

from engines.standard_engines import SupervisedTrainer as DefaultSupervisedTrainer
from engines.standard_engines import SupervisedEvaluator as DefaultSupervisedEvaluator 

from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys

#################################################### Imports for computing all of the validation modes, e.g. Dice computation etc.
import csv 
from monai.handlers import MeanDice 
from monai.metrics import DiceHelper, DiceMetric, do_metric_reduction 
from monai.transforms import Activations, AsDiscrete 
from monai.utils import MetricReduction
import copy

from datetime import datetime

logger = logging.getLogger(__name__)


def run(self_dict, 
        engine: DefaultSupervisedTrainer | DefaultSupervisedEvaluator,
        batchdata: dict[str, torch.Tensor]) -> dict:


    if self_dict['train']:

        #We have no inner loop, this is a default segmentation network with no interactivity component.
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
      
        return engine._iteration(engine=engine, batchdata=batchdata)  # type: ignore[arg-type]

    else:

        #We have no inner loop, this is a default segmentation network with no interactivity component.


        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

    
        return engine._iteration(engine=engine, batchdata=batchdata)  # type: ignore[arg-type]