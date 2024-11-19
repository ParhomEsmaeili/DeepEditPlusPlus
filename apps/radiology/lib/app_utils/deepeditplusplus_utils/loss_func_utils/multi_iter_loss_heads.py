
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Optional
import torch
import re

from operator import mul
from functools import reduce 
#Importing the interactive version..
from engines.interactive_seg_engines.trainer import SupervisedTrainer as InteractiveSupervisedTrainer
from engines.interactive_seg_engines.evaluator import SupervisedEvaluator as InteractiveSupervisedEvaluator 



'''
A wrapper intended for use in generating losses using multi-iteration loss heads. Designed to be used for both standard losses and masked losses.

It is also designed to be compatible with different components in the loss: Per Iteration Loss, Cross Iteration Loss

This version is NOT compatible with deep supervision outputs.
'''

class MultiIterationLossHeads():
    
    def __init__(self, 
                # mask_strategy: dict[str, dict[str, str]], 
                memory_length: str, 
                loss_class_init_dict: dict[str,dict], 
                reduction: str, 
                per_iter_weight_bool: bool,
                version_param: str,
                per_iter_weight_strategy: Optional[dict[str, float]] = None,
                intra_iter_weight_dict: Optional[dict[str, dict[str, dict[str, float]]]] = None,
                
                ):

        '''
        Args:

        memory_length: This is a string which dictates the window which is used for the loss heads. This can be Final (only the final iteration) or Full (all iterations) 

        loss_class_init_dict: This is a nested dict which contains the class initialisations for each subcomponent in the components of the loss, in the 
        following structure: 

        -Mode : (Autoseg initialisation, Interactive Initialisation, Interactive Editing)
            sub-component (e.g. local responsiveness, temporal consistency, base)
                -class init.


        These class initialisations should be initialised with a parametrisation of any clicks (if FIXED). Else, this will be passed through in the click info. If fixed,
        #then the parametrisation dict from click info will be Nonetype. 

        reduction: The reduction strategy across the multi-iteration loss heads (e.g. sum or mean)

        per_iter_weight_bool: The bool which determines whether there is a uniform or non-uniform weighting across the loss heads.
        
        per_iter_weight_strategy: An optional dict which contains which determines what strategy is being used to generate the weights across the loss heads for the 
        non-uniform weighting, in addition to any parametrisation.

        intra_iter_weight_dict : An optional dict which contains relative weights within each iteration's loss head between the sub-components in the loss (e.g. local loss, temporal consistency etc)
        
        contains an initial weight for the mode, and an iterative weight if applicable (for the editing mode). These all fall under a key:val pair where the key
        is the strategy used for generating the weights.

        structure:
        -Mode : (Autoseg initialisation, Interactive Initialisation, Interactive Editing)
            sub-component (e.g. local responsiveness, temporal consistency, base)
                - dict containing fields: initial value, per iteration addition (can be None for the initialisation modes)



        version_param: The str which determines which version of this class is being used for performing the "forward" computation.
        '''

        # self.mask_strategies = mask_strategy 
        self.memory_length = memory_length 
        self.loss_class_init_dict = loss_class_init_dict 
        self.reduction = reduction 
        self.per_iter_weight_bool = per_iter_weight_bool

        self.per_iter_weight_strategy_str = per_iter_weight_strategy
        self.intra_iter_weight_dict = intra_iter_weight_dict
        self.version_param = version_param

        # supported_mask_strategies = ['None',
        #                             'Ellipsoid',
        #                             'Exponentialised Scaled Euclidean Distance']
        
        supported_memory_length = ['Final', 
                                    'Full']
        
        supported_reduction = ['Sum',
                            'Mean',
                            'Weighted Sum',
                            'Weighted Mean'
                            ]
        
        supported_version_params = ['1']

        supported_per_iter_weight_strategies = [
                                                ]

        supported_intra_iter_weight_strategies = [
                                                ]


        if self.version_param not in supported_version_params:
            raise ValueError('The version parameter selected was not supported.')
        
        # for component_dict in self.mask_strategies.values():

        #     if any([i not in supported_mask_strategies for i in list(component_dict.values())]):
        #         raise ValueError("The mask strategy was not supported.")

        if self.reduction.title() not in supported_reduction:
            raise ValueError("The selected cross-iter reduction strategy was not supported")
        
        if not self.per_iter_weight_strategy_str == None:

            if self.per_iter_weight_strategy_str not in supported_iter_weight_strategies:
                raise ValueError("The selected per iter weight strategy was not supported")

        if not self.intra_iter_weight_dict == None:

            if any([i not in suported_intra_iter_weight_strategies for i in self.intra_iter_weight_dict.keys()]):
                raise ValueError("The selected intra iter weight strategy was not supported")

    def data_synthesis_config(self, 
                            click_info: dict[str, dict[str, dict[str, dict]]]
                            )-> dict[str, bool | str | int]:
        '''
        Args:

        Click info dict which contains all of the click information for each sample in the batch across all of the inference/edit iterations for that batch.
        '''

        #Supported Initialisation names:

        supported_init_names = ['Interactive Init', 'Autoseg Init']

        #We assert that the names of the iterations must be consistent across the entire batch:
        
        for sample_index_name, sample_click_info in click_info.items():
            
            if sample_index_name == 'sample_index_1':
                iter_names = list(sample_click_info.keys())
            else:
                if list(sample_click_info.keys()) != iter_names:
                    raise ValueError('There were inconsistencies between the batch in the subloop performed in the inner loop')
        
        output_config_dict = dict() 

        if len(iter_names) == 1:
            #In this case, the subloop was just an initialisation.
            initialisation_name = iter_names[0]

            if initialisation_name.title() not in supported_init_names:
                raise ValueError("The initialisation name was not valid")

            output_config_dict['Editing Bool'] = False
            output_config_dict['Initialisation Name'] = initialisation_name.title()
            
        else:
            #In this case, the subloop was initialisation + editing
            
            initialisation_name = iter_names[0]

            if initialisation_name.title() not in supported_init_names:
                raise ValueError("The initialisation name was not valid")
            
    
            output_config_dict['Editing Bool'] = True 
            output_config_dict['Initialisation Name'] = initialisation_name.title()
            output_config_dict['Num Editing Iters'] = len(iter_names) - 1


        return output_config_dict

    def per_iter_weight_generator(self, num_iterations: int) -> list[float]:
        '''Function which computes the weighting for multi-iteration loss heads:

            Args:
            Num_iterations: Number of iterations in the inner loop including the initialisation.
        '''

        if self.per_iter_weight_bool:

            # raise NotImplementedError("The function is not implemented for generating the non-uniform weights for multi-iteration loss heads")
            assert len(list(self.per_iter_weight_strategy.keys())) == 1

            if list(self.per_iter_weight_strategy.keys())[0] == 'Add':
                
                if self.per_iter_weight_strategy['Add'] <= 0:
                    raise ValueError('Per iter additional weighting should be > 0')
                
                full_list = [1.0 + i * self.per_iter_weight_strategy['Add'] for i in range(num_iterations)]

                if self.memory_length == 'Full':
                    return full_list
                elif self.memory_length == 'Final':
                    return [full_list[-1]]
                
            elif list(self.per_iter_weight_strategy.keys()[0]) == 'Multiply':
                
                if self.per_iter_weight_strategy['Multiply'] < 1:
                    raise ValueError('Per iter additional weighting on the multiplier should be > 1 (additional penalty)')

                full_list = [self.per_iter_weight_strategy['Multiply'] ** i for i in range(num_iterations)]

                if self.memory_length == 'Full':
                    return full_list
                
                elif self.memory_length == 'Final':
                    return [full_list[-1]]
        else:

            if self.memory_length == 'Full':

                return [1.0] * num_iterations

            elif self.memory_length == 'Final':

                return [1.0]
    def intra_iter_loss_component_weight_generator(self, data_synthesis_config: dict[str, bool | str | int]) -> dict[str, dict[str, float]]:
        '''
        Function which generates the weighting on the loss function components for each iteration in the data synthesis subloop.

        Args:

        data_synthesis_config: The configuration that was used in the inner loop for generating the data used for performing optimisation: consists of the following fields:

        #Editing_bool: True/False which controls whether interactive editing was used.
        #Initialisation Name: The name of the initialisation strategy that was used.
        #Num Editing Iters: The number of editing iterations in the subloop (if interactive editing was used)

        Outputs:

        dictionary which contains the weightings of the loss function components for each iteration. It is structured as follows

        - Iteration name
            - Loss function component (e.g. base, local responsiveness, temporal consistency)
                - weighting parameter, a float value, should be no less than 0.
        '''
        if self.intra_iter_weight_dict != None:
            
            '''This subloop is intended to generate the weights for each of the components for each mode/iteration using the initial val and per iteration multiplier/addition param'''
            raise NotImplementedError("The function for generating non-uniform intra-iter weights was not implemented (between subcomponents/components in the loss)")

        else:

            #In this case, the weighting is assumed to be uniform across all components for each mode regardless of the subcomponents.
            intra_iter_weight_dict = dict()

            if data_synthesis_config['Editing Bool']:
                #If editing was performed:

                if self.memory_length == 'Final':

                

                    #Extracting the final editing iter names and setting the values for each of the components in that mode configuration.
                    
                        
                    intra_iter_weight_dict[f'Editing Iter {data_synthesis_config["Num Editing Iters"]}'] = dict()

                    for component in self.loss_class_init_dict['Interactive Editing'].keys():
                        intra_iter_weight_dict[f'Editing Iter {data_synthesis_config["Num Editing Iters"]}'][component] = 1.0  

                elif self.memory_length == 'Full':

                    #Extracting the initialisation name and setting the values for each of the components 

                    intra_iter_weight_dict[data_synthesis_config['Initialisation Name']] = dict()

                    for component in self.loss_class_init_dict[data_synthesis_config['Initialisation Name']].keys():
                        intra_iter_weight_dict[data_synthesis_config['Initialisation Name']][component] = 1.0 

                    

                    #Extracting the editing iter names and setting the values for each of the components in that mode configuration.
                    for index in range(data_synthesis_config['Num Editing Iters']):
                        
                        intra_iter_weight_dict[f'Editing Iter {index + 1}'] = dict()

                        for component in self.loss_class_init_dict['Interactive Editing'].keys():
                            intra_iter_weight_dict[f'Editing Iter {index + 1}'][component] = 1.0  


            else:
                #Extracting the initialisation name and setting the values for each of the components 

                intra_iter_weight_dict[data_synthesis_config['Initialisation Name']] = dict()
                
                for component in self.loss_class_init_dict[data_synthesis_config['Initialisation Name']].keys():
                    intra_iter_weight_dict[data_synthesis_config['Initialisation Name']][component] = 1.0 


            return intra_iter_weight_dict

    def single_iter_loss_computation(self,
                        click_set: dict,
                        click_parametrisation: dict,
                        preds_dict: dict[str, torch.Tensor],
                        targets: torch.Tensor,
                        data_synthesis_mode: str,
                        loss_component_weighting:dict[str, float]
                        ) -> torch.Tensor:
        '''
        This function performs the batch-wise loss computation for a single iteration (and use mode). The loss is averaged across the batch (and across voxels) for each
        subcomponent. The losses are summed across the subcomponents and weighted according to their weighting parameter in the loss component weighting dict.

        Args: 

        Click sets: A dictionary which contains the sets of clicks for the corresponding iteration (or lackthereof for the autoseg initialisation) across the batch
        Click parametrisations: A dictionary which contains the parametrisation of the clicks if they are dynamic (else the parametrisation is passed through in the
        initialisation of the loss component/function class.) across the batch.
        preds_dict: The dictionary consisting of the batchwise predictions for both the current (and optionally the prior iteration's) predictions.
        targets: The batchwise ground truth labels provided in a single tensor format.
        data_synthesis_mode: The mode in which the synthesised data presented for the current iteration was generated
        loss_component_weighting: The dictionary of weights which parametrise the weighting of the loss subcomponents for the given iteration.


        '''

        supported_subcomponent_types = ['Base', 'Local Responsiveness', 'Temporal Consistency']

        #Extracting the dictionary of loss computation subcomponent classes. 

        loss_computation_subcomponents = self.loss_class_init_dict[data_synthesis_mode.title()]

        per_component_loss_values_list = []
        
        for subcomponent_name, subcomponent_class in loss_computation_subcomponents.items():
            
            #Checking that these are all supported! 
            if subcomponent_name not in supported_subcomponent_types:
                raise ValueError("A selected loss function component was not  of a supported type")

            #Assumed format is that for the base, it only takes the current iteration's prediction, and the ground truth targets.
            if subcomponent_name == "Base":
                
                unweighted_loss = subcomponent_class(preds_dict=preds_dict, targets=targets).mean() #Should be a single value already from the reduction method selected in the
                #base of the loss component.
                per_component_loss_values_list.append(loss_component_weighting[subcomponent_name] * unweighted_loss)

            #Assumed format is that for the local reponsiveness loss, that it takes the current iteration's prediction, ground truth targets, and the click info
            #from the current iteration. 

            elif subcomponent_name == "Local Responsiveness":
                
                unweighted_loss = subcomponent_class(preds_dict=preds_dict, targets=targets, click_sets = click_set, click_parametrisations = click_parametrisation).mean()
                #Should be a single value already from the reduction method selected in the base of the loss component.
                per_component_loss_values_list.append(loss_component_weighting[subcomponent_name] * unweighted_loss)

            #Assumed format is that for the temporal consistency loss, that it takes the current and prior iteration's predictions, the ground truth target, and the 
            #click info from the current iteration. 

            elif subcomponent_name == "Temporal Consistency":

                unweighted_loss = subcomponent_class(preds_dict=preds_dict, targets=targets, click_sets = click_set, click_parametrisations = click_parametrisation).mean()
                #Should be a single value already from the reduction method selected in the #base of the loss component, mean is not really necessary here..
                per_component_loss_values_list.append(loss_component_weighting[subcomponent_name] * unweighted_loss)
            

        
        return sum(per_component_loss_values_list) #Returning the weighted sum of the loss components.
            
    def multi_iter_loss_computation_handler(self, 
                                engine, 
                                click_info: dict[str, dict[str, dict]], 
                                inner_pred_inputs: dict[str, dict[str, torch.Tensor]], #These are the inputs required in order to make preds in train mode
                                inner_loop_preds: dict[str, dict[str, torch.Tensor]], #These are the predictions that were made in eval mode during the inner loop (for the "GT" of temporal consistency computations)
                                config_dict: dict[str, bool | int | str],
                                final_inputs: torch.Tensor, 
                                targets: torch.Tensor,
                                loss_component_weighting: dict[str, dict[str, float]]) -> list:

        batch_size = len(click_info.keys())

        if self.memory_length == 'Full':

            #It is necessary to use and output this because the final prediction is used for computing the train dice metric.
            final_preds = engine.inferer(final_inputs.detach(), engine.network)
        

            if not config_dict['Editing Bool']:

                #If it is not editing then it is initialisation only:
                # we only pass through the predictions of the given final iteration.

                preds_dict = {'Current Preds': final_preds}

                loss_component_weights = loss_component_weighting[config_dict['Initialisation Name']]
            
                data_synthesis_mode = config_dict['Initialisation Name']
                
                click_sets = dict() 
                click_parametrisations = dict() 

                for sample_index in range(batch_size):

                    click_sets[f'sample_index_{sample_index + 1}'] = click_info[f'sample_index_{sample_index + 1}'][data_synthesis_mode]['click_set']
                    click_parametrisations[f'sample_index_{sample_index + 1}'] = click_info[f'sample_index_{sample_index + 1}'][data_synthesis_mode]['click_parametrisation']

                loss_list = [self.single_iter_loss_computation(
                    click_sets,
                    click_parametrisations,
                    preds_dict,
                    targets,
                    data_synthesis_mode,
                    loss_component_weights
                    )]
            else:

                #If it is init + editing:
                inner_loop_iters_keys = list(click_info['sample_index_1'].keys())

                loss_list = []

                for iteration_index, inner_loop_iter in enumerate(inner_loop_iters_keys):
                    
                    if iteration_index == 0:
                        if inner_loop_iter != 'Interactive Init' and inner_loop_iter != 'Autoseg Init':
                            raise KeyError('The first key was not for an initialisation mode')
                    else:
                        if inner_loop_iter != f'Editing Iter {iteration_index}':
                            raise KeyError('The key did not correspond to the correct index in the editing iterations/ or was not in the correct order.')  

    
                    if iteration_index == 0:
                        loss_component_weights = loss_component_weighting[config_dict['Initialisation Name']]
                        data_synthesis_mode = config_dict['Initialisation Name']


                        inputs_list = []
                        # prior_preds_list = []
                        for sample_index in range(batch_size):

                            sample_index_string = f'sample_index_{sample_index + 1}'
                            
                            #Input was from the initialisation iter.
                            inputs_list.append(inner_pred_inputs[sample_index_string][f'{data_synthesis_mode} Input'].detach())
                            

                        #Placing the inputs onto the cuda device in batch format (BHWD).
                        inputs =  torch.stack(inputs_list).cuda()
                        
                        #Generating predictions, output is in the format BNHWD (B = batch size, N = number of classes.)
                        current_preds = engine.inferer(inputs, engine.network)
                        
                        #Given that it is an initialisation, prior predictions are unnecessary.
                        preds_dict = {'Current Preds': current_preds}


                        #Extracting the click sets and parametrisations for the init iteration under consideration
                        click_sets = dict() 
                        click_parametrisations = dict() 

                        for sample_index in range(batch_size):
                            
                            sample_index_string = f'sample_index_{sample_index + 1}'

                            #If initialisation, then extract the corresponding appropriate click sets and parametrisations using the name of the initialisation.
                            click_sets[sample_index_string] = click_info[sample_index_string][data_synthesis_mode]['click_set']

                            click_parametrisations[sample_index_string] = click_info[sample_index_string][data_synthesis_mode]['click_parametrisation']                   
                
                        #Computing the loss from the single iteration head.
                        loss_list.append(self.single_iter_loss_computation(
                            click_sets,
                            click_parametrisations,
                            preds_dict,
                            targets,
                            data_synthesis_mode,
                            loss_component_weights
                            ))
                        
                        #We then extract the eval mode computed predictions for the prior predictions (necessary for temporal consistency computation)

                        
                        prior_preds_list = []
                        for sample_index in range(batch_size):

                            sample_index_string = f'sample_index_{sample_index + 1}'
                            
                            prior_preds_list.append(inner_loop_preds[sample_index_string][f'{data_synthesis_mode} Pred'].detach())
                           
                        #Placing the prior preds onto the cuda device in batch format (BHWD). #NOTE: This is in B1HWD format, same as the ground truth label.
                        prior_predictions = torch.stack(prior_preds_list).cuda()

                    else:
                        #If the iteration is an editing iteration:
                        loss_component_weights = loss_component_weighting[f'Editing Iter {iteration_index}']

                        data_synthesis_mode = 'Interactive Editing'

                        #If the temporal consistency loss component is in the interactive editing mode loss term:

                        if 'Temporal Consistency' in self.loss_class_init_dict[data_synthesis_mode].keys():

                            if iteration_index == len(inner_loop_iters_keys) - 1:
                                #In this scenario, the current prediction would be the FINAL set of predictions
                                preds_dict = {'Current Preds': final_preds, 'Prior Preds': prior_predictions}

                            else:
                                #If it is not the final editing iteration, then we need to compute the current prediction.
                                
                                
                                inputs_list = []
                                
                                for sample_index in range(batch_size):

                                    sample_index_string = f'sample_index_{sample_index + 1}'
                                    
                                    #Input was from the editing iter.
                                    inputs_list.append(inner_pred_inputs[sample_index_string][f'Editing Iter {iteration_index} Input'].detach())

                                #Placing the inputs onto the cuda device in batch format (BHWD).
                                inputs =  torch.stack(inputs_list).cuda()
                                
                                #Generating predictions, output is in the format BNHWD (B = batch size, N = number of classes.)
                                current_preds = engine.inferer(inputs, engine.network)
                                
                                
                                preds_dict = {'Current Preds': current_preds, 'Prior Preds': prior_predictions}


                                #We then extract the eval mode computed predictions for the prior predictions (necessary for temporal consistency computation)
                                #This is done by extracting the prediction for the CURRENT iteration. (Would be the prior one for the next iteration.)

                                prior_preds_list = []
                                for sample_index in range(batch_size):

                                    sample_index_string = f'sample_index_{sample_index + 1}'
                                    
                                    prior_preds_list.append(inner_loop_preds[sample_index_string][f'Editing Iter {iteration_index} Pred'].detach())
                                
                                #Placing the prior preds onto the cuda device in batch format (B1HWD).
                                prior_predictions = torch.stack(prior_preds_list).cuda()

                        else:
                            #If not, then temporal consistency is not under consideration and so prior predictions are unnecessary.

                            if iteration_index == len(inner_loop_iters_keys) - 1:
                                #In this scenario, the current prediction would be the FINAL set of predictions
                                preds_dict = {'Current Preds': final_preds}

                            else:
                                #If it is not the final editing iteration, then we need to compute the current prediction.

                                inputs_list = []
                                
                                for sample_index in range(batch_size):

                                    sample_index_string = f'sample_index_{sample_index + 1}'
                                    
                                    #Input was from the editing iter.
                                    inputs_list.append(inner_pred_inputs[sample_index_string][f'Editing Iter {iteration_index} Input'].detach())

                                #Placing the inputs onto the cuda device in batch format (BHWD).
                                inputs =  torch.stack(inputs_list).cuda()
                                
                                #Generating predictions, output is in the format BNHWD (B = batch size, N = number of classes.)
                                current_preds = engine.inferer(inputs, engine.network)
                                
                                preds_dict = {'Current Preds': current_preds}


                        #Extracting the click sets and parametrisations for the current iteration under consideration
                        click_sets = dict() 
                        click_parametrisations = dict() 

                        for sample_index in range(batch_size):
                            
                            sample_index_string = f'sample_index_{sample_index + 1}'

                           
                            #If editing:
                            click_sets[sample_index_string] = click_info[sample_index_string][f'Editing Iter {iteration_index}']['click_set']

                            click_parametrisations[sample_index_string] = click_info[sample_index_string][f'Editing Iter {iteration_index}']['click_parametrisation'] 

                
                        #Computing the loss from the single iteration head.
                        loss_list.append(self.single_iter_loss_computation(
                            click_sets,
                            click_parametrisations,
                            preds_dict,
                            targets,
                            data_synthesis_mode,
                            loss_component_weights
                            ))
                # engine.fire_event(IterationEvents.FORWARD_COMPLETED) #Currently there is no function that actually does anything with this, intended to execute with the 
                #prediction/forward pass (but we have multiple!! so we leave this disabled for now.)

        elif self.memory_length == 'Final':
            
            #It is necessary to use and output this because the final prediction is used for computing the train dice metric.
            final_preds = engine.inferer(final_inputs.detach(), engine.network)

            #We pass through a predictions dictionary which contains the batchwise prediction for the current (final) and the batchwise predictions for the prior iter, 
            #if it exists (i.e. if the mode used was an editing mode)

            if config_dict['Editing Bool']:


                #If true, then we may need to pass through the prior iteration's set of predictions in addition to the final set.
                
                loss_component_weights = loss_component_weighting[f'Editing Iter {config_dict["Num Editing Iters"]}']

                data_synthesis_mode = 'Interactive Editing'

                #If the temporal consistency loss component is in the interactive editing mode loss term:

                if 'Temporal Consistency' in self.loss_class_init_dict[data_synthesis_mode].keys():

                    # #First we accumulate across the batch for the given iteration under consideration for the prior inputs. 
                    # prior_inputs_list = []
                    # for sample_index in range(batch_size):

                    #     sample_index_string = f'sample_index_{sample_index + 1}'
                        
                    #     if config_dict['Num Editing Iters'] > 1:
                    #         #If num editing iters > 1, then prior iter was an editing iter
                    #         prior_inputs_list.append(inner_pred_inputs[sample_index_string][f'Editing Iter {config_dict["Num Editing Iters"] - 1} Input'])
                    #     else:
                    #         #Else: it was the initialisation iter.
                    #         prior_inputs_list.append(inner_pred_inputs[sample_index_string][f'{config_dict["Initialisation Name"]} Input'])

                    # #Placing the prior inputs onto the cuda device in batch format (BHWD).
                    # prior_inputs =  torch.stack(prior_inputs_list).cuda()
                    
                    # #Generating prior predictions, output is in the format BNHWD (B = batch size, N = number of classes.)
                    # prior_predictions = engine.inferer(prior_inputs, engine.network)

                    #We extract the eval mode computed predictions for the prior prediction (necessary for temporal consistency computation)
                    #This is done by extracting the prediction for the CURRENT iteration. (Would be the prior one for the next iteration.)

                    prior_preds_list = []
                    for sample_index in range(batch_size):

                        sample_index_string = f'sample_index_{sample_index + 1}'

                        if config_dict['Num Editing Iters'] > 1:
                            #If num editing iters > 1, then prior iter was an editing iter
                            prior_preds_list.append(inner_loop_preds[sample_index_string][f'Editing Iter {config_dict["Num Editing Iters"] - 1} Pred'].detach())
                        else:
                            #Else: it was the initialisation iter.
                            prior_preds_list.append(inner_loop_preds[sample_index_string][f'{config_dict["Initialisation Name"]} Pred'].detach())
                        

                    #Placing the prior preds onto the cuda device in batch format (BHWD).
                    prior_predictions = torch.stack(prior_preds_list).cuda()

                    preds_dict = {'Current Preds': final_preds, 'Prior Preds': prior_predictions}
                else:
                    #If not, then temporal consistency is not under consideration and so prior predictions are unnecessary.
                    preds_dict = {'Current Preds': final_preds}


                #Extracting the click sets and parametrisations for the current iteration under consideration
                click_sets = dict() 
                click_parametrisations = dict() 

                for sample_index in range(batch_size):
                    
                    sample_index_string = f'sample_index_{sample_index + 1}'
                    click_sets[sample_index_string] = click_info[sample_index_string][f'Editing Iter {config_dict["Num Editing Iters"]}']['click_set']

                    click_parametrisations[sample_index_string] = click_info[sample_index_string][f'Editing Iter {config_dict["Num Editing Iters"]}']['click_parametrisation']
                
            

            else:
                #If not, then it was an initialisation mode, and we only pass through the predictions of the given final iteration.

                preds_dict = {'Current Preds': final_preds}

                loss_component_weights = loss_component_weighting[config_dict['Initialisation Name']]
            
                data_synthesis_mode = config_dict['Initialisation Name']
                
                click_sets = dict() 
                click_parametrisations = dict() 

                for sample_index in range(batch_size):

                    click_sets[f'sample_index_{sample_index + 1}'] = click_info[f'sample_index_{sample_index + 1}'][data_synthesis_mode]['click_set']
                    click_parametrisations[f'sample_index_{sample_index + 1}'] = click_info[f'sample_index_{sample_index + 1}'][data_synthesis_mode]['click_parametrisation']

            #Computing the loss from the single iteration head.
            loss_list = [self.single_iter_loss_computation(
                click_sets,
                click_parametrisations,
                preds_dict,
                targets,
                data_synthesis_mode,
                loss_component_weights
                )]

            # engine.fire_event(IterationEvents.FORWARD_COMPLETED) #Currently there is no function that actually does anything with this, intended to execute with the 
            #prediction/forward pass (but we have multiple!! so we leave this disabled for now.)

        return final_preds, loss_list





    def multi_iteration_reduction(self, 
                                per_iteration_losses: list[torch.Tensor], per_iteration_weights: list[float]):
        '''
        Args:

        per_iteration_losses: The batch-wise averaged loss heads at each iteration in the multi-iteration loss.
        per_iteration_weights: The weight for each loss head.


        '''
        if self.per_iter_weight_bool:
            #Non-uniform weighting across per-iteration loss heads. 
            
            supported_non_uniform_reductions = ['Weighted Sum', 'Weighted Mean']
            if self.reduction.title() not in supported_non_uniform_reductions:
                raise ValueError('Not a compatible reduction strategy')
            
            #Applying weighting to the loss terms: 

            weighted_losses = [x * y for x,y in zip(per_iteration_losses, per_iteration_weights)]
            
            if self.reduction.title() == 'Weighted Sum':
                reduced_loss = sum(weighted_losses)
            
            elif self.reduction.title() == 'Weighted Mean':

                summed_weights = sum(per_iteration_weights)
                reduced_loss = sum(weighted_losses)/summed_weights
        else:
            #Uniform weighting
            supported_uniform_reductions = ['Sum', 'Mean']
            if self.reduction.title() not in supported_uniform_reductions:
                raise ValueError('Not a compatible reduction strategy')
            
            if self.reduction.title() == 'Sum':
                reduced_loss = sum(per_iteration_losses)
            
            elif self.reduction.title() == 'Mean':
                reduced_loss = sum(per_iteration_losses) / len(per_iteration_losses)
                
        return reduced_loss 


    def __call__(self, 
                engine: InteractiveSupervisedTrainer | InteractiveSupervisedEvaluator, 
                click_info: dict[str, dict[str, dict]], 
                inner_pred_inputs: dict[str, dict[str, torch.Tensor]], #These are the inputs that were used for generating those.
                inner_loop_preds: dict[str, dict[str, torch.Tensor]],  #These are predictions made in eval mode that are discretised.
                final_inputs: torch.Tensor, 
                targets: torch.Tensor):
        
        # NOTE: Each batches' output in the loss function should output a single tensor value (like a float). 

        if self.version_param == '1':
            
            #Extracting the configuration in the inner loop which has been used to generate the synthesised data:
            config_dict = self.data_synthesis_config(click_info)

            #Extracting the weightings for each of the loss components for each mode & for each iter
            loss_component_weighting = self.intra_iter_loss_component_weight_generator(config_dict)

            
            final_preds, per_iter_losses = self.multi_iter_loss_computation_handler(engine, click_info, inner_pred_inputs, inner_loop_preds, config_dict, final_inputs, targets, loss_component_weighting)
            
            per_iter_weights = self.per_iter_weight_generator(num_iterations=len(click_info['sample_index_1']))
            overall_loss = self.multi_iteration_reduction(per_iter_losses, per_iter_weights)

            return final_preds, overall_loss

