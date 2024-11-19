'''
File which contains the function which produces the loss function class dictionaries with click-based masks.. 

Version = 1: Multi-iter loss heads for base loss component only.
Version = 2: Multi-iteration loss heads with locality component also.
Version = 3: Multi=iteration loss heads with temporal consistency component also (on top of version 2)
'''

import logging

import torch

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

# from monai.losses import DiceCELoss 
from loss_func_utils.multi_iter_loss_heads import MultiIterationLossHeads
from loss_func_utils.generalised_masked_loss_wrapper import GeneralisedMaskedLossWrapper

logger = logging.getLogger(__name__)

def run(mask_strategies: dict[str, list], #The dictionary containing the list of masking strategies for a given component.
        foundation_loss_config: dict[str, dict], #The dictionary containing the dictionary for the raw loss (to be wrapped) #for each component (fixed across use modes for a given component, e.g. base or local responsiveness)
        generalised_mask_wrapper_config: dict, #The dictionary containing the information required for initialising the generalised masked loss wrapper.
        multi_iter_config: dict, #The dictionary containing the information required for initialising the multi-iter class wrapper.
        click_param_types_config: dict[str,str], #The dictionary containing the type of click parametrisation for a given component (e.g. None, Fixed, Dynamic)
        click_param_values_config: dict[str, list], #The dictionary containing the click parametrisation for the masks of a given component (e.g. base, locality)
        subcomponents: dict[str, list], #A dictionary containing the use modes, and the list of corresponding components for each use mode.
        func_version_param:str):

    supported_func_version_params = ['1', '2', '3']

    supported_use_modes = ['Autoseg Init', 'Interactive Init', 'Interactive Editing']

    supported_subcomponents = ['Base', 'Local Responsiveness', 'Temporal Consistency']

    unique_subcomponents = set()

    for subcomponent_list in subcomponents.values():
        unique_subcomponents.update(subcomponent_list)
    
    if func_version_param not in supported_func_version_params: 
        raise ValueError('This version of the function is not yet supported')
    if any([i not in supported_use_modes for i in subcomponents.keys()]):
        raise ValueError('At least one of the use modes selected in generating loss classes is not supported')
    if any([i not in supported_subcomponents for i in unique_subcomponents]):
        raise ValueError('At least one of the selected subcomponents is not supported')

    if func_version_param == '1':

        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.

        mask_strategy = dict()
        #TODO: Modify the mask strategy dict when we add the locality and temporal consistency based losses.
       
        mask_strategy['Autoseg Init'] = {'Base': mask_strategies['Base']}
        mask_strategy['Interactive Init'] = {'Base':mask_strategies['Base']} 
        mask_strategy['Interactive Editing'] = {'Base':mask_strategies['Base']} 

        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = dict() 

        for mode in subcomponents.keys():

            click_parametrisation_type[mode] = dict() 
            for subcomponent in subcomponents[mode]:
                click_parametrisation_type[mode][subcomponent] = click_param_types_config[subcomponent]
    
        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation = dict() 

        for mode in subcomponents.keys():
        #Initialising each mode:
            click_parametrisation[mode] = dict()
            for subcomponent in subcomponents[mode]:
                click_parametrisation[mode][subcomponent] = click_param_values_config[subcomponent]
        
        #For now we will assume that the foundation of the loss is consistent across the modes for each subcomponent 

        subcomponent_loss_foundation_config = foundation_loss_config 

        #Initialising the loss computation classes.

        loss_class_init_dict = dict()

        for mode in mask_strategy.keys():
            
            loss_class_init_dict[mode] = dict()

            for subcomponent in mask_strategy[mode].keys():

                loss_class_init_dict[mode][subcomponent] = GeneralisedMaskedLossWrapper(
                                                        click_parametrisation_type=click_parametrisation_type[mode][subcomponent],
                                                        click_parametrisation_info=click_parametrisation[mode][subcomponent],
                                                        mask_strategy=mask_strategy[mode][subcomponent],
                                                        loss_config_dict=subcomponent_loss_foundation_config[subcomponent],
                                                        subcomponent_name=subcomponent) 
        
        return MultiIterationLossHeads(
            multi_iter_config['memory_length'], # Final = Only use the final iteration inputs-outputs to compute the loss. Full = Use All.
            loss_class_init_dict,
            multi_iter_config['reduction'], #The multi-iteration loss reduction method.
            multi_iter_config['per_iter_weight_bool'], #The per-iteration weighting strategy for the multi-iteration loss heads.
            multi_iter_config['version_param'] #The version of "__call__" we are using from the MultiIterationLossHeads class.

        )
    

    if func_version_param == '2':

        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.

        mask_strategy = dict()
        mask_strategy['Autoseg Init'] = {'Base': mask_strategies['Base']}
        mask_strategy['Interactive Init'] = {'Base':mask_strategies['Base'],
                                            'Local Responsiveness': mask_strategies['Local Responsiveness']}
        mask_strategy['Interactive Editing'] = {'Base':mask_strategies['Base'], 
                                                'Local Responsiveness': mask_strategies['Local Responsiveness']} #'Temporal Consistency': temporal_consist_mask_strategy}

        
        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = dict() 

        for mode in subcomponents.keys():

            click_parametrisation_type[mode] = dict() 
            for subcomponent in subcomponents[mode]:
                click_parametrisation_type[mode][subcomponent] = click_param_types_config[subcomponent]
    
        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation = dict() 

        for mode in subcomponents.keys():
        #Initialising each mode:
            click_parametrisation[mode] = dict()
            for subcomponent in subcomponents[mode]:
                click_parametrisation[mode][subcomponent]  = click_param_values_config[subcomponent]
        
        #For now we will assume that the foundation of the loss is consistent across the modes for each subcomponent 

        subcomponent_loss_foundation_config = foundation_loss_config 
        
        #Initialising the loss computation classes.

        loss_class_init_dict = dict()

        for mode in mask_strategy.keys():
            
            loss_class_init_dict[mode] = dict()

            for subcomponent in mask_strategy[mode].keys():

                loss_class_init_dict[mode][subcomponent] = GeneralisedMaskedLossWrapper(
                                                        click_parametrisation_type=click_parametrisation_type[mode][subcomponent],
                                                        click_parametrisation_info=click_parametrisation[mode][subcomponent],
                                                        mask_strategy=mask_strategy[mode][subcomponent],
                                                        loss_config_dict=subcomponent_loss_foundation_config[subcomponent],
                                                        subcomponent_name=subcomponent,
                                                        masked_wrapper_reduction= generalised_mask_wrapper_config['Batch Reduction'][subcomponent]) 
     
        return MultiIterationLossHeads(
            multi_iter_config['memory_length'],
            loss_class_init_dict,
            multi_iter_config['reduction'],
            multi_iter_config['per_iter_weight_bool'],
            multi_iter_config['version_param'],

        )

    if func_version_param == '3':

        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.

        mask_strategy = dict()
        mask_strategy['Autoseg Init'] = {'Base': mask_strategies['Base']}
        mask_strategy['Interactive Init'] = {'Base':mask_strategies['Base'],
                                            'Local Responsiveness': mask_strategies['Local Responsiveness']}
        mask_strategy['Interactive Editing'] = {'Base':mask_strategies['Base'], 
                                                'Local Responsiveness': mask_strategies['Local Responsiveness'],
                                                'Temporal Consistency': mask_strategies['Temporal Consistency']}

        
        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = dict() 

        for mode in subcomponents.keys():

            click_parametrisation_type[mode] = dict() 
            for subcomponent in subcomponents[mode]:
                click_parametrisation_type[mode][subcomponent] = click_param_types_config[subcomponent]
    
        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation = dict() 

        for mode in subcomponents.keys():
        #Initialising each mode:
            click_parametrisation[mode] = dict()
            for subcomponent in subcomponents[mode]:
                click_parametrisation[mode][subcomponent]  = click_param_values_config[subcomponent]
        
        #For now we will assume that the foundation of the loss is consistent across the modes for each subcomponent 

        subcomponent_loss_foundation_config = foundation_loss_config 
        
        #Initialising the loss computation classes.

        loss_class_init_dict = dict()

        for mode in mask_strategy.keys():
            
            loss_class_init_dict[mode] = dict()

            for subcomponent in mask_strategy[mode].keys():

                loss_class_init_dict[mode][subcomponent] = GeneralisedMaskedLossWrapper(
                                                        click_parametrisation_type=click_parametrisation_type[mode][subcomponent],
                                                        click_parametrisation_info=click_parametrisation[mode][subcomponent],
                                                        mask_strategy=mask_strategy[mode][subcomponent],
                                                        loss_config_dict=subcomponent_loss_foundation_config[subcomponent],
                                                        subcomponent_name=subcomponent,
                                                        masked_wrapper_reduction= generalised_mask_wrapper_config['Batch Reduction'][subcomponent]) 
     
        return MultiIterationLossHeads(
            multi_iter_config['memory_length'],
            loss_class_init_dict,
            multi_iter_config['reduction'],
            multi_iter_config['per_iter_weight_bool'],
            multi_iter_config['version_param'],
        )
        