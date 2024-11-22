'''
File which contains the function which produces the loss function component in the train setup. 
Version = -1: Default Dice Cross Entropy loss with deep supervision loss heads
Version = 0: Default DeepEdit loss, an unweighted Dice - Cross Entropy loss. 
Version = 1: Multi-iteration loss heads, reduction = sum, unweighted Dice - Cross Entropy loss for each use mode and head. FULL iteration memory for loss heads.
Version = 2: Multi-iteration loss heads, reduction = sum, unweighted dice - cross entropy loss for each use mode and head. FINAL iteration memory only for loss heads.

Version = 3: Multi-iteration loss heads, reduction = sum, ONLY the memory of the final iteration's loss head. unweighted DiceCE for autoseg head, locality component
also for the interactive modes.
Version = 4: Multi-iteration loss heads, reduction = sum, only the memory of the final iteration's loss head. unweighted dicece for autoseg head, locality 
for interactive init, and locality + temporal consistency for the editing heads..

Version = 5: Multi-iteration loss heads, reduction = sum, full memory of per iteration's loss head. unweighted DiceCE for autoseg head, locality component
also for the interactive modes.

Version = 6: Multi-iteration loss heads, reduction = sum, full memory of per iteration's loss head. unweighted DiceCE for autoseg head, locality 
for interactive init, and locality + temporal consistency for the editing heads..
'''

import logging

import torch

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

# from monai.losses import DiceCELoss 
from loss_func_utils.ds_loss import DeepSupervisionLoss
from loss_func_utils.dice import DiceCELoss 
from loss_func_utils.multi_iter_loss_heads import MultiIterationLossHeads
from loss_func_utils.generalised_masked_loss_wrapper import GeneralisedMaskedLossWrapper
from loss_func_utils.multi_iteration_loss_class_generator import run

logger = logging.getLogger(__name__)

def run_get_loss_func(self_dict, context, func_version_param):

    assert type(self_dict) == dict 
    assert type(func_version_param) == str 

    supported_version_params = ['-1', '0', '1', '2', '3', '4']
    
    assert func_version_param in supported_version_params, 'The version parameter was not supported for obtaining the loss function'

    if func_version_param == '-1':
        #This verison is intended for a standard segmentation engine, where the final set of inputs from an inner loop are used to perform a standard forward
        #pass for performing a global segmentation based loss computation.

        return DeepSupervisionLoss(DiceCELoss(to_onehot_y=True, softmax=True))

    elif func_version_param == '0':
        #This version is intended for a standard segmentation engine, where the final set of inputs from an inner loop are used to perform a forward pass
        #for computing the loss on a global segmentation only.
        return DiceCELoss(to_onehot_y=True, softmax=True)


    if func_version_param == '1':
        
        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.
        subcomponents_dict = {
            'Autoseg Init': ['Base'],
            'Interactive Init': ['Base'],
            'Interactive Editing': ['Base']
        }

        
        mask_strategies = dict()
        
        mask_strategies['Base'] = ['None']  
        
        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = {
            'Base': 'None'
        }

        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation_vals = {
            'Base': None #Base does not require parametrisation
        }

        #Defining the inputs for initialising the classes for each of the loss components. For now we will assume that these are consistent across the modes for each
        #subcomponent 

        subcomponent_loss_foundation_config = dict()

        subcomponent_loss_foundation_config['Base'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':True,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #Multi-iteration class wrapper config:

        multi_iter_config = dict()

        multi_iter_config['memory_length'] = 'Full' # Final = Only use the final iteration inputs-outputs to compute the loss. Full = Use All
        multi_iter_config['reduction'] = 'Sum'  #The multi-iteration loss reduction method.
        multi_iter_config['per_iter_weight_bool'] = False #The per-iteration weighting strategy for the multi-iteration loss heads.
        multi_iter_config['version_param'] = '1' #The version of "__call__" we are using from the MultiIterationLossHeads class.
        
        return run(
            mask_strategies=mask_strategies,
            foundation_loss_config=subcomponent_loss_foundation_config,
            generalised_mask_wrapper_config=dict(),
            multi_iter_config=multi_iter_config,
            click_param_types_config=click_parametrisation_type,
            click_param_values_config=click_parametrisation_vals,
            subcomponents=subcomponents_dict,
            func_version_param='1'
            )
    
    if func_version_param == '2':
        
        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.
        subcomponents_dict = {
            'Autoseg Init': ['Base'],
            'Interactive Init': ['Base'],
            'Interactive Editing': ['Base']
        }

        
        mask_strategies = dict()
        
        mask_strategies['Base'] = ['None']  
        
        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = {
            'Base': 'None'
        }

        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation_vals = {
            'Base': None #Base does not require parametrisation
        }

        #Defining the inputs for initialising the classes for each of the loss components. For now we will assume that these are consistent across the modes for each
        #subcomponent 

        subcomponent_loss_foundation_config = dict()

        subcomponent_loss_foundation_config['Base'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':True,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #Multi-iteration class wrapper config:

        multi_iter_config = dict()

        multi_iter_config['memory_length'] = 'Final' # Final = Only use the final iteration inputs-outputs to compute the loss. Full = Use All
        multi_iter_config['reduction'] = 'Sum'  #The multi-iteration loss reduction method.
        multi_iter_config['per_iter_weight_bool'] = False #The per-iteration weighting strategy for the multi-iteration loss heads.
        multi_iter_config['version_param'] = '1' #The version of "__call__" we are using from the MultiIterationLossHeads class.
        
        return run(
            mask_strategies=mask_strategies,
            foundation_loss_config=subcomponent_loss_foundation_config,
            generalised_mask_wrapper_config=dict(),
            multi_iter_config=multi_iter_config,
            click_param_types_config=click_parametrisation_type,
            click_param_values_config=click_parametrisation_vals,
            subcomponents=subcomponents_dict,
            func_version_param='1'
            )
    

    if func_version_param == '3':

        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.
        subcomponents_dict = {
            'Autoseg Init': ['Base'],
            'Interactive Init': ['Base', 'Local Responsiveness'],
            'Interactive Editing': ['Base', 'Local Responsiveness']
        }

        
        temporal_consist_mask_strategy = ['Ellipsoid']
        
        mask_strategies = dict()
        
        mask_strategies['Base'] = ['None']
        mask_strategies['Local Responsiveness'] = ['Ellipsoid']

        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = {
            'Base': 'None',
            'Local Responsiveness': 'Fixed'
        }

        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation_vals = {
            'Base': None, #Base does not require parametrisation,
            'Local Responsiveness': {'Ellipsoid':[5,5,5]},
        }

        
        #Defining the inputs for initialising the classes for each of the loss components. For now we will assume that these are consistent across the modes for each
        #subcomponent 

        subcomponent_loss_foundation_config = dict()

        subcomponent_loss_foundation_config['Base'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':True,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #TODO: Implement the ones for the non-base components when we add the locality and temporal consistency support.

        subcomponent_loss_foundation_config['Local Responsiveness'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':False,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #Initialising the config for the generalised mask loss wrapper:

        generalised_mask_wrapper_config = dict()

        #Setting the reduction strategy for the batch-wise computation (for the maskedlosswrapper).
        generalised_mask_wrapper_config['Batch Reduction'] = {
            'Base': None,
            'Local Responsiveness': 'Mean'
        }

        
        #Multi-iteration class wrapper config:

        multi_iter_config = dict()

        multi_iter_config['memory_length'] = 'Final' # Final = Only use the final iteration inputs-outputs to compute the loss. Full = Use All
        multi_iter_config['reduction'] = 'Sum'  #The multi-iteration loss reduction method.
        multi_iter_config['per_iter_weight_bool'] = False #The per-iteration weighting strategy for the multi-iteration loss heads.
        multi_iter_config['version_param'] = '1' #The version of "__call__" we are using from the MultiIterationLossHeads class.
        
        return run(
            mask_strategies=mask_strategies,
            foundation_loss_config=subcomponent_loss_foundation_config,
            generalised_mask_wrapper_config=generalised_mask_wrapper_config,
            multi_iter_config=multi_iter_config,
            click_param_types_config=click_parametrisation_type,
            click_param_values_config=click_parametrisation_vals,
            subcomponents=subcomponents_dict,
            func_version_param='2'
            )
    

    if func_version_param == '4':

        #Setting the mask strategy for each mode and their corresponding loss components. For now we assume the same type of loss components have the same mask strategy
        #regardless of the mode.
        subcomponents_dict = {
            'Autoseg Init': ['Base'],
            'Interactive Init': ['Base', 'Local Responsiveness'],
            'Interactive Editing': ['Base', 'Local Responsiveness', 'Temporal Consistency']
        }

        mask_strategies = dict()
        
        mask_strategies['Base'] = ['None']
        mask_strategies['Local Responsiveness'] = ['Ellipsoid']
        mask_strategies['Temporal Consistency'] = ['Ellipsoid'] 

        #Setting the click parametrisation types for the masks:
        click_parametrisation_type = {
            'Base': 'None',
            'Local Responsiveness': 'Fixed',
            'Temporal Consistency': 'Fixed'
        }

        #Setting the parametrisation for the masks, if applicable. 
        click_parametrisation_vals = {
            'Base': None, #Base does not require parametrisation,
            'Local Responsiveness': {'Ellipsoid':[5,5,5]},
            'Temporal Consistency': {'Ellipsoid':[5,5,5]}
        }

        #Defining the inputs for initialising the classes for each of the loss components. For now we will assume that these are consistent across the modes for each
        #subcomponent 

        subcomponent_loss_foundation_config = dict()

        subcomponent_loss_foundation_config['Base'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':True,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #TODO: Implement the ones for the non-base components when we add the locality and temporal consistency support.

        subcomponent_loss_foundation_config['Local Responsiveness'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':False,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        subcomponent_loss_foundation_config['Temporal Consistency'] = {
            'name': 'DiceCELoss',
            
            #The actual params required.
            'include_background': True,
            'to_onehot_y':False,
            'sigmoid':False,
            'softmax':True,
            'other_act':None,
            'squared_pred':False,
            'jaccard':False,
            'reduction':"mean",
            'smooth_nr':1e-5,
            'smooth_dr':1e-5,
            'batch':False,
            'weight':None,
            'lambda_dice':1.0,
            'lambda_ce':1.0,
            'label_smoothing':0.0,
        }

        #Initialising the config for the generalised mask loss wrapper:

        generalised_mask_wrapper_config = dict()

        #Setting the reduction strategy for the batch-wise computation (for the maskedlosswrapper).
        generalised_mask_wrapper_config['Batch Reduction'] = {
            'Base': None,
            'Local Responsiveness': 'Mean',
            'Temporal Consistency': 'Mean'
        }

        
        #Multi-iteration class wrapper config:

        multi_iter_config = dict()

        multi_iter_config['memory_length'] = 'Final' # Final = Only use the final iteration inputs-outputs to compute the loss. Full = Use All
        multi_iter_config['reduction'] = 'Sum'  #The multi-iteration loss reduction method.
        multi_iter_config['per_iter_weight_bool'] = False #The per-iteration weighting strategy for the multi-iteration loss heads.
        multi_iter_config['version_param'] = '1' #The version of "__call__" we are using from the MultiIterationLossHeads class.
        
        return run(
            mask_strategies=mask_strategies,
            foundation_loss_config=subcomponent_loss_foundation_config,
            generalised_mask_wrapper_config=generalised_mask_wrapper_config,
            multi_iter_config=multi_iter_config,
            click_param_types_config=click_parametrisation_type,
            click_param_values_config=click_parametrisation_vals,
            subcomponents=subcomponents_dict,
            func_version_param='3'
            )
    

        