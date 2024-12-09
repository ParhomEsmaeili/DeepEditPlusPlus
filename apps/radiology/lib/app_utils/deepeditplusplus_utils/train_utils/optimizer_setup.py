'''
File which contains the functions which produces the optimization strategy in the train setup. 
'''

import logging

import torch
from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)
from monai_handlers import (
    LrScheduleHandler,
    MeanDice,
    from_engine
)


logger = logging.getLogger(__name__)

def run_get_train_optimizer(self_dict, context, func_version_param):

    '''Versions:
    
    Version 0: The default deepedit optimiser (ADAM) with initial learning rate of 0.0001 (fixed)
    Version 1: An optimiser with a selectable initial learning rate for the ADAM optimiser.
    '''
    assert type(self_dict) == dict 
    assert type(func_version_param) == str 

    supported_version_params = ['0','1']

    if func_version_param not in supported_version_params:
        raise ValueError('This version of the optimiser generator is not supported.')
    
    if func_version_param == '0':
        return torch.optim.Adam(context.network.parameters(), lr=0.0001)
    
    elif func_version_param == '1':
        return torch.optim.Adam(context.network.parameters(), lr=self_dict['component_parametrisation_dict']['init_lr'])

def run_get_train_lr_scheduler(self_dict, context, func_version_param):

    '''
    Versions:

    Version 0: A learning rate scheduler which adjusts the lr according to the step size (1000 epochs) by a factor of 0.1
    
    '''
    assert type(self_dict) == dict
    assert type(func_version_param) == str 

    supported_version_params = ['0']

    if func_version_param not in supported_version_params:
        raise ValueError('This version of the lr handler is not supported')

    if func_version_param == '0':
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(context.optimizer, mode="min")
        # return LrScheduleHandler(lr_scheduler, print_lr=True, step_transform=lambda x: x.state.output[0]["loss"])

        lr_scheduler = torch.optim.lr_scheduler.StepLR(context.optimizer, step_size=1000, gamma=0.1)
        return LrScheduleHandler(lr_scheduler, print_lr=True)
     