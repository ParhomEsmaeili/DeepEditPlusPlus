'''
Generalised masked loss wrapper as the Monai codebase does not contain any masked wrapper which is not intended for a discrete mask.

For non-masked subcomponents in the loss it just generates a tensor of ones anyways.
'''

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Optional
import torch
import re

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

#Importing the interactive version..
from engines.interactive_seg_engines.trainer import SupervisedTrainer as InteractiveSupervisedTrainer
from engines.interactive_seg_engines.evaluator import SupervisedEvaluator as InteractiveSupervisedEvaluator 

from loss_func_utils.mask_generator_utils_cpu import MaskGenerator
from loss_func_utils.dice import DiceCELoss as RawDiceCELoss
from loss_func_utils.spatial_mask import MaskedLoss 

from monai.networks.utils import one_hot

class GeneralisedMaskedLossWrapper:
    
    def __init__(self,
                click_parametrisation_type: str,
                click_parametrisation_info: dict[str, list] | None,
                mask_strategy: list[str],
                loss_config_dict: dict,
                subcomponent_name: str,
                masked_wrapper_reduction: Optional[str] = None):

        '''Args:
            
            click_parametrisation_type: The string containing the type of the click parametrisation, None, Fixed, Dynamic.
            click_parametrisation_info: The parametrisation info for the subcomponent of the loss function for each mask type (allows for combinations).
            
            Structure is either a dict, with each key being the mask_strategy name, and the value being the correpsonding parametrisation: The parameterisation 
            is a list for fixed parametrisation of click for each mask type. 
            
            OR, the click_parametrisation_info can be Nonetype for dynamic click parametrisation/non parametrised loss func, e.g. a non click-based mask weighting). 
            For dynamic click parametrisation, this needs to be fed through the inner loop.

            mask_strategy: This is a list which contains the mask strategy names for the given component (e.g. Ellipsoid, Exponentialised Scaled Euclidean Distance, None)
            It MUST be length 1, we do not currently permit a combination of these masks.. 

            loss_config_dict: The name and input parameters for defining the base of the loss component (e.g. the dice loss beneath the masked wrapper).

            subcomponent_name: The name of the subcomponent, e.g. Base, Local Responsiveness, Temporal Consistency 
            
            masked_wrapper_reduction: The name of the batch-wise reduction strategy for the masked loss wrapper (each item in batch must be passed through)
            individually to the loss computation because number of voxels might vary according to the mask.

        '''

        supported_mask_strategies = ['None',
                                    'Ellipsoid',
                                    'Cuboid',
                                    'Binarised Exponentialised Scaled Euclidean Distance']

        supported_foundation_names = ['DiceCELoss',
                                ]

        supported_subcomponent_names = ['Base',
                                        'Local Responsiveness',
                                        'Temporal Consistency']

        supported_masked_wrapper_reductions = [
                                            'Mean',
                                            'Sum']

        if len(mask_strategy) != 1:
            raise ValueError("The number of strategies provided in the mask strategy list was not correct, should be 1.")

        if any([i not in supported_mask_strategies for i in mask_strategy]):
            raise ValueError("A mask strategy was not supported.")

        loss_name = loss_config_dict['name']

        if loss_name not in supported_foundation_names:
            raise ValueError("The name of the base of the loss was not one of those that are valid/supported")
        
        if subcomponent_name not in supported_subcomponent_names:
            raise ValueError("The name of the subcomponent was not one of those that are valid/supported")

        if masked_wrapper_reduction not in supported_masked_wrapper_reductions and masked_wrapper_reduction != None:
            raise ValueError("The batch-wise reduction strategy in the masked wrapper is not valid/supported.")

        if masked_wrapper_reduction == None:
            if subcomponent_name != 'Base':
                raise ValueError('There was an invalid batchwise reduction strategy selected for a non-base subcomponent.')
        #Initialising the subcomponent name:
        self.subcomponent_name = subcomponent_name

        #Initialising the click parametrisation type name:
        self.click_parametrisation_type = click_parametrisation_type


        #Initialising the mask generator class:
        self.mask_strategy = mask_strategy 

        self.initialise_mask_generator_class() 

        #Initialising the loss computation class:
        self.masked_wrapper_reduction = masked_wrapper_reduction
        self.loss_comp_call = self.initialise_loss_comp_class(loss_config_dict)

        #Setting the click parametrisation info:

        self.click_parametrisation_info = click_parametrisation_info

    def initialise_mask_generator_class(self):
        '''
        Returns the class used for generating masks using the given mask strategy.
        '''
        if self.subcomponent_name.title() == 'Base':
            self.mask_generator_class = None

        elif self.subcomponent_name.title() == 'Local Responsiveness':
            human_measure = 'Local Responsiveness' #This should usually be equivalent to the name of the subcomponent..
            self.mask_generator_class = MaskGenerator(self.mask_strategy, human_measure)

        elif self.subcomponent_name.title() == 'Temporal Consistency':
            human_measure = 'Temporal Consistency'
            self.mask_generator_class = MaskGenerator(self.mask_strategy, human_measure)

    def initialise_raw_loss_comp(self, name:str, loss_config_dict: dict):
        'Generates the non click-based weighted loss computation class according to the given name'

        if name == 'DiceCELoss':

            return RawDiceCELoss(
                include_background = loss_config_dict['include_background'],
                to_onehot_y = loss_config_dict['to_onehot_y'],
                sigmoid = loss_config_dict['sigmoid'],
                softmax = loss_config_dict['softmax'],
                other_act = loss_config_dict['other_act'],
                squared_pred = loss_config_dict['squared_pred'],
                jaccard = loss_config_dict['jaccard'],
                reduction = loss_config_dict['reduction'],
                smooth_nr = loss_config_dict['smooth_nr'],
                smooth_dr = loss_config_dict['smooth_dr'],
                batch = loss_config_dict['batch'],
                weight = loss_config_dict['weight'],
                lambda_dice = loss_config_dict['lambda_dice'],
                lambda_ce = loss_config_dict['lambda_ce'],
                label_smoothing = loss_config_dict['label_smoothing'],
            )

    def initialise_loss_comp_class(self, loss_config_dict):

        raw_loss_name = loss_config_dict['name']
        
        if self.subcomponent_name == 'Base':

            #We assert that the click parametrisation type must be None:
            if not self.click_parametrisation_type == 'None':
                raise ValueError('Base component should have no click parametrisation/click mask type')
            #In this case just initialise the default loss functions
            return self.initialise_raw_loss_comp(raw_loss_name, loss_config_dict)

        elif self.subcomponent_name == 'Local Responsiveness':
            
            #We assert that the click parametrisation type must NOT be None:
            if self.click_parametrisation_type == 'None':
                raise ValueError('Local responsiveness component should not have click parametrisation/mask type of None')

            #We assert that the foundation of the loss configuration is appropriate for the masked loss wrapper implemented:
            if loss_config_dict['to_onehot_y'] == True:
                raise ValueError('There should be no one-hot splitting of the targets for the foundation loss class in the local responsiveness component')
            

            #Extracting the raw loss class (i.e. what we will be wrapping)
            raw_loss_class = self.initialise_raw_loss_comp(raw_loss_name, loss_config_dict)

            #Wrapping it:
            # raise NotImplementedError('There may be an issue with the masked loss, it does not ignore the outside regions, it just sets them to zero, so mean.reduction() will not work as it should')
            return MaskedLoss(raw_loss_class, self.masked_wrapper_reduction)
            
        elif self.subcomponent_name == 'Temporal Consistency':
            
            #We assert that the click parametrisation type must NOT be None:
            if self.click_parametrisation_type == 'None':
                raise ValueError('Temporal consistency component should not have click parametrisation/mask type of None')
             
            #Extracting the raw loss class (i.e. what we will be wrapping)
            raw_loss_class = self.initialise_raw_loss_comp(raw_loss_name, loss_config_dict)

            #Wrapping it:
            # raise NotImplementedError('There may be an issue with the masked loss, it does not ignore the outside regions, it just sets them to zero, so mean.reduction() will not work as it should')
            return MaskedLoss(raw_loss_class, self.masked_wrapper_reduction)

    def generate_click_guidance_parametrisations(self, 
                                                guidance_points_set: Optional[dict[str, dict[str, list]] | None] = None, 
                                                guidance_points_parametrisations: Optional[dict[str, dict[str, dict[str, list]]] | None] = None):

        if self.click_parametrisation_type == 'None':

            #First we assert that the guidance points set should be nonetype:
            if guidance_points_set != None:
                raise ValueError('The guidance points set should be NoneType for click_parametrisation type: NONE')

            #In this case, there is no parametrisation at all (i.e. no mask was required.) We must assert this however!
            

            if self.click_parametrisation_info != None:
                raise ValueError('There should be no parametrisation for instances where there click parametrisation type is None!')
            if guidance_points_parametrisations != None:
                raise ValueError('There should be no parametrisation for the guidance points!')

        elif self.click_parametrisation_type == 'Dynamic':
            #In this case, the parametrisation should've already been passed through/provided, we must still assert however!

            raise NotImplementedError('Requires a handler for instances where there is an empty click set for a class, needs to produce a dummy parametrisation..?')
            # #First we assert click set is not None type:
            # if guidance_points_set == None:
            #     raise ValueError('Guidance points set is NoneType even though it should not be!')
            
            # #We then make assertions on the parametrisations being available for each mask type.
            # for mask_type in self.mask_strategy:
            #     #We will assert that every mask type in the mask_strategy has a parametrisation in the parametrisation dictionary..
            #     if mask_type not in guidance_points_parametrisations['sample_index_1'].keys():
            #         raise KeyError(f'The parameters for the mask type {mask_type} was required, but was not provided.')
                
        elif self.click_parametrisation_type == 'Fixed':
            
            if guidance_points_parametrisations != None:
                raise ValueError('There should be no existing parametrisation of the guidance points')

            guidance_points_parametrisations = dict() 

            for sample_index in guidance_points_set.keys():
                
                guidance_points_parametrisations[sample_index] = dict() 

                for mask_type in self.mask_strategy:
                    #Extracting the list of parametrisation values:
                    parametrisations_list = self.click_parametrisation_info[mask_type]

                    if type(parametrisations_list) != list:
                        raise ValueError('The parametrisations for the fixed parametrisation were not presented in a list!') 
                    
                    
                    current_mask_type_dict = dict() 

                    for class_label, nested_list_of_points in guidance_points_set[sample_index].items(): 
                        
                        #Initialise the nested list for the current class label: 
                        if len(nested_list_of_points) > 0:
                            class_wide_params_list = [parametrisations_list] * len(nested_list_of_points)
                            # Multiplied by the length of the nested list of points for each class. Fixed parametrisation across all clicks!sat                            class_wide_params_list = [parametrisations] * len(nested_list_of_points) 

                        else:
                            #There may be instances where specific click sets are empty for certain classes. In this situation we need a dummy parametrisation. 
                            class_wide_params_list = [parametrisations_list]

                        current_mask_type_dict[class_label] = class_wide_params_list 

                        guidance_points_parametrisations[sample_index][mask_type] = current_mask_type_dict 

            return guidance_points_parametrisations

    def generate_mask(self, guidance_points_set: dict[str, dict[str, list]], guidance_points_parametrisations: dict[str,dict[str, dict[str, list]]], image_dims: torch.Size):
        
        masks_list = []
        
        for sample_index in guidance_points_set.keys():

            sample_mask = self.mask_generator_class(guidance_points_set[sample_index], guidance_points_parametrisations[sample_index], True, image_dims)
            #Need to reshape it to be 1NHWD from HWD. 
            sample_mask = torch.unsqueeze(sample_mask, dim=0) 

            masks_list.append(sample_mask)
        
        mask = torch.stack(masks_list)

        
        #Need to check that the mask is a binarised mask, we check if the elements are not 0 or 1
        
        invalid_mask = ~torch.isin(mask, torch.tensor([0,1]))
        if invalid_mask.sum() > 0:
            raise ValueError('The loss mask must be binarised!')
        else:
            return mask.cuda()
    
    def split_gt(self, pred:torch.Tensor, target: torch.Tensor):#, background_last: bool):
        
        'Splits the target label by class in one hot format, required for the masked loss implementation..'

        n_ch = pred.shape[1]
        assert target.shape[1] == 1
        
        if n_ch == 1:
            warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
        else:
            target = one_hot(target, num_classes=n_ch)
        
        return target

    def __call__(self,
                preds_dict: dict[str, torch.Tensor], 
                targets: torch.Tensor, 
                click_sets:Optional[dict[str, dict[str, list]]] = None, 
                click_parametrisations: Optional[str, dict[str, list]] = None):
        '''
        Args:

        preds_dict: The dictionary containing the set of predictions under consideration. This could be just the current iteration's predictions, or it could be the 
        prior also, for example (e.g. for temporal consistency computation). This is across the batch.

        Assumption is that the predictions are not discretised! The loss functions always assume that the predictions are not discretised, and that the targets
        are discretised.


        targets: The ground truth labels across the batch (in BNHWD or B1HWD shape).

        click_sets: The dictionary of click sets across the batch, with sample indices in the same order as the preds and targets' batch order in their tensors.
        click_parametrisations: The dictionary of click parametrisations across the batch, with sample indices in the same order as the preds and targets' batch order.
        
        Returns:

        The loss component value averaged across the batch provided.
        '''

        #Obtaining the loss function call:

        if self.click_parametrisation_type == 'None':
            
            #Extracting the masks: If click parametrisation_type is None then there is not any masking! There is also no requirement for generating a 
            #click parametrisaion dict.

            
            #There is no scenario where a locality based loss has no click based weightmask, but there may be for a temporal consistency component which 
            #just works using foundational/basic loss component between previous pred and the current pred. 

            if self.subcomponent_name == 'Base':
                
                #Run it through the guidance points checker:
                self.generate_click_guidance_parametrisations()
                #If base then it uses current predictions and the targets.

                return self.loss_comp_call(preds_dict['Current Preds'], targets)

        elif self.click_parametrisation_type == 'Fixed':
            
            # raise NotImplementedError("The masked losses are not yet implemented")
            click_parametrisations = self.generate_click_guidance_parametrisations(click_sets)
            mask = self.generate_mask(click_sets, click_parametrisations, targets.shape[2:])
            #The mask generator was already initialised with the human measure/subcomponent name so no need to split it here..

            if self.subcomponent_name == 'Local Responsiveness':
                #split target by class to get BNHWD 
                targets = self.split_gt(preds_dict['Current Preds'], targets)

                return self.loss_comp_call(preds_dict['Current Preds'], targets, mask)
            
            elif self.subcomponent_name == 'Temporal Consistency':

                #split prior pred by class to get BNHWD
                split_prior_pred = self.split_gt(preds_dict['Current Preds'], preds_dict['Prior Preds'])

                return self.loss_comp_call(preds_dict['Current Preds'], split_prior_pred, mask)

        elif self.click_parametrisation_type == 'Dynamic':   
            
            # raise NotImplementedError('The masked losses are not yet implemented.')

            self.generate_click_guidance_parametrisations()
            
            mask = self.generate_mask(click_sets, click_parametrisations, targets.shape[2:])
            #The mask generator was already initialised with the human measure/subcomponent name so no need to split it here..

            if self.subcomponent_name == 'Local Responsiveness':
                #split target by class to get BNHWD 
                targets = self.split_gt(preds_dict['Current Preds'], targets)

                return self.loss_comp_call(preds_dict['Current Preds'], targets, mask)
            
            elif self.subcomponent_name == 'Temporal Consistency':

                #split prior pred by class to get BNHWD
                split_prior_pred = self.split_gt(preds_dict['Current Preds'], preds_dict['Prior Preds'])

                return self.loss_comp_call(preds_dict['Current Preds'], split_prior_pred, mask)
        

             