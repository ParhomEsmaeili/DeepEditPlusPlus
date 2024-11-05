'''
Version of the inner loop which is deepedit++ like but performs variable number of editing iterations prior to the final iteration being inputted to the default engine.

INTENDED FOR THE DEFAULT VERSION. 

Does not extract or save any information about the click sets for the click based loss functions!

Also changed the click probability to = 1 (no dropout). Will require some approach for penalising a lack of sufficiently quick convergence potentially!

For validation, the full set of editing iterations is implemented.
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

engines_dir = os.path.join(up(up(up(up(up(up(up(os.path.abspath(__file__)))))))), 'engines')
sys.path.append(engines_dir)

from engines.standard_engines import SupervisedTrainer as DefaultSupervisedTrainer
from engines.standard_engines import SupervisedEvaluator as DefaultSupervisedEvaluator 


#Importing the interactive version..
from engines.interactive_seg_engines import SupervisedTrainer as InteractiveSupervisedTrainer
from engines.interactive_seg_engines import SupervisedEvaluator as InteractiveSupervisedEvaluator 

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
        engine: DefaultSupervisedTrainer | DefaultSupervisedEvaluator | InteractiveSupervisedTrainer | InteractiveSupervisedEvaluator,
        batchdata: dict[str, torch.Tensor]) -> dict:

    if self_dict['train']:

        #In this case, the inner loop is just being used for training.
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        label_names = batchdata["label_names"]
        
        if np.random.choice([True, False], p=[self_dict['interactive_init_probability'], 1 - self_dict['interactive_init_probability']]):
            #Here we run the loop for Interactive initialisation (which is the default)
            logger.info("Interactive Init. Inner Subloop")
            
        else:
            #Here we run the loop for generating autoseg input data by zeroing out the input click channels
            batchdata_list = decollate_batch(batchdata, detach=True)
            for k in range(len(batchdata_list)):
                for i in range(self_dict['num_intensity_channel'], self_dict['num_intensity_channel'] + len(label_names)):
                    batchdata_list[k][CommonKeys.IMAGE][i] *= 0
            batchdata = list_data_collate(batchdata_list)
            logger.info("AutoSegmentation Inner Subloop")
            
        #Here we print whether the input is on the cuda device from the initialisation:
        logger.info(f'The input image and label directly after the initialisation: Image is on cuda: {batchdata["image"].is_cuda}, Label is on cuda: {batchdata["label"].is_cuda}')

        #Here we use the prior input fields to generate a prediction (our previous seg) and generate a new set of inputs with this updated previous seg.
        
        if np.random.choice([True, False], p  = [self_dict['deepedit_probability'], 1 - self_dict['deepedit_probability']]):
        
            logger.info("Interactive Editing mode Inner Subloop")

            #We randomly generate the number of editing iterations being performed.
            max_iterations = np.random.randint(1, self_dict['max_iterations'] + 1) # + 1 because randint is not inclusive of the upper val.

            for j in range(max_iterations):
                inputs, _ = engine.prepare_batch(batchdata)
                #Next line puts the inputs on the cuda device
                inputs = inputs.to(engine.state.device)
                
                engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                engine.network.eval()

                #Printing which device the image is on prior to the inner loop prediction.
                logger.info(f'The input image prior to the inner loop inference: Image is on cuda: {inputs.is_cuda}')

                with torch.no_grad():
                    logger.info(f'The model prior to the inner loop inference is on device {next(engine.network.parameters()).device}') 
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            #Runs the inferer on the cuda device
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)
                batchdata.update({CommonKeys.PRED: predictions})
                
                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)
                
                #Checking whether pred, Image metatensor and label metatensor are on cuda device here:
                logger.info(f'The pre-click transform inputs: Image is on cuda: {batchdata_list[0]["image"].is_cuda}, Label is on cuda {batchdata_list[0]["label"].is_cuda}, Prediction is on cuda {batchdata_list[0]["pred"].is_cuda}')
               
                for i in range(len(batchdata_list)):
                    batchdata_list[i][self_dict['click_probability_key']] = 1.0 
                 
                    batchdata_list[i] = self_dict['transforms'](batchdata_list[i])

                #############################################################

                batchdata = list_data_collate(batchdata_list)
                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        
        # first item in batch only
        engine.state.batch = batchdata

        logger.info(f'For the final inputs the image is on cuda: {batchdata["image"].is_cuda}, the label is on cuda: {batchdata["label"].is_cuda}')
        logger.info(f'For the engine, amp is {engine.amp}')
        
        return engine._iteration(engine, batchdata)  # type: ignore[arg-type]

    else:

        #In this case, the inner loop is being used for validation.


        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        label_names = batchdata["label_names"]
        
        ######################## Code for creating dir for saving output files ##############################
        # meta_dict_filename = batchdata["image"].meta["filename_or_obj"][0]
        # studies_name = meta_dict_filename.split('/')[meta_dict_filename.split('/').index('datasets') + 1]
        
        external_val_output_dir = self_dict['external_validation_output_dir'] #, 'external_validation', studies_name)

        if not os.path.exists(external_val_output_dir):
            os.makedirs(external_val_output_dir)

        ############ Implementation of validation in-the-loop so that we can validate across all modes. ######################
        #We just do autoseg and interactive init for now, deepedit (the with an automatic pre-transform one) is already done as part of the default validation config).
        
        if not self_dict['train']:
            
            logger.info('Validation occuring')
            #Make a diff copy of the params e.g. batchdata so it does not affect the normal validation. 
            
            ######################################### THESE VALIDATION METHODS ONLY APPLY FOR BATCH SIZE 1 ######################################
            batchdata_val_deepgrow = copy.deepcopy(batchdata)
            #First we compute the interactive init mode validation generations:
            inputs_val_deepgrow, _ = engine.prepare_batch(batchdata_val_deepgrow)
            inputs_val_deepgrow = inputs_val_deepgrow.to(engine.state.device)

            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_deepgrow = engine.inferer(inputs_val_deepgrow, engine.network)
                else:
                    predictions_val_deepgrow = engine.inferer(inputs_val_deepgrow, engine.network)
            



            #Now we do autoseg based mode validations generation:
            batchdata_val_autoseg = copy.deepcopy(batchdata)

            batchdata_list_val_autoseg = decollate_batch(batchdata_val_autoseg, detach=True)
            for i in range(self_dict['num_intensity_channel'], self_dict['num_intensity_channel'] + len(label_names)):
                batchdata_list_val_autoseg[0][CommonKeys.IMAGE][i] *= 0
            batchdata_val_autoseg = list_data_collate(batchdata_list_val_autoseg)
            
            inputs_val_autoseg, _ = engine.prepare_batch(batchdata_val_autoseg)
            inputs_val_autoseg = inputs_val_autoseg.to(engine.state.device)

            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_autoseg = engine.inferer(inputs_val_autoseg, engine.network)
                else:
                    predictions_val_autoseg = engine.inferer(inputs_val_autoseg, engine.network)
            
            
            
            ################# Process the predictions and label for metric computation ###########################
            discretised_label = AsDiscrete(argmax=False, to_onehot=len(label_names))(batchdata['label'][0])

            #Interactive init.
            deepgrow_activation = Activations(softmax=True)(predictions_val_deepgrow[0])
            deepgrow_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(deepgrow_activation)
            
            #We will only compute the meandice for now, no need for dice on each individual label currently.., 
            # just need to see how it stacks up comparatively to the DeepEdit one (any fluctuation between classes likely shows up on DeepEdit/Editing mode also)


            #Autoseg
            autoseg_activation = Activations(softmax=True)(predictions_val_autoseg[0])
            autoseg_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(autoseg_activation)



            #Metric computations:

            deepgrow_dice = DiceHelper(  # type: ignore
                include_background= False,
                sigmoid = False,
                softmax = False, 
                activate = False,
                get_not_nans = False,
                reduction = MetricReduction.MEAN, #MetricReduction.MEAN,
                ignore_empty = True,
                num_classes = None
                )(y_pred=deepgrow_discretised_pred.cpu().unsqueeze(dim=0), y=discretised_label.unsqueeze(dim=0))
            
            autoseg_dice = DiceHelper(  # type: ignore
                include_background= False,
                sigmoid = False,
                softmax = False, 
                activate = False,
                get_not_nans = False,
                reduction = MetricReduction.MEAN, #MetricReduction.MEAN,
                ignore_empty = True,
                num_classes = None
                )(y_pred=autoseg_discretised_pred.cpu().unsqueeze(dim=0), y=discretised_label.unsqueeze(dim=0))
            

        ###################################################### Normal implementation of DeepEdit++ #########################################################################

        if np.random.choice([True, False], p=[self_dict['interactive_init_probability'], 1 - self_dict['interactive_init_probability']]):
            #Here we run the loop for Interactive Initialisation (which is the default)
            logger.info("Interactive Init. Inner Subloop")
            
        
        else:
            #Here we run the loop for generating autoseg prompt channels by zeroing out the click channels.

            batchdata_list = decollate_batch(batchdata, detach=True)
            for k in range(len(batchdata_list)):
                for i in range(self_dict['num_intensity_channel'], self_dict['num_intensity_channel'] + len(label_names)):
                    batchdata_list[k][CommonKeys.IMAGE][i] *= 0
            batchdata = list_data_collate(batchdata_list)
            logger.info("AutoSegmentation Inner Subloop")
            
        #Here we print whether the input is on the cuda device from the initialisation:
        logger.info(f'The input image and label directly after the initialisation: Image is on cuda: {batchdata["image"].is_cuda}, Label is on cuda: {batchdata["label"].is_cuda}')

        #Here we use the initial input data to generate a prediction (our new previous seg) and generate a new set of inputs with this updated previous seg.
        
        if np.random.choice([True, False], p  = [self_dict['deepedit_probability'], 1 - self_dict['deepedit_probability']]):
        
            logger.info("Editing mode Inner Subloop")
            for j in range(self_dict['max_iterations']):
                inputs, _ = engine.prepare_batch(batchdata)
                #Next line puts the inputs on the cuda device
                inputs = inputs.to(engine.state.device)
                
                # if not self.train:
                #     if not os.path.exists(os.path.join(output_dir, 'TrainingInnerLoop')):
                        
                #         os.makedirs(os.path.join(output_dir, 'TrainingInnerLoop'))

                #     batchdata_list = decollate_batch(batchdata, detach=True)
                #     for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                #         placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                #         placeholder = np.array(placeholder_tensor[i])
                #         #print(placeholder)
                #         nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoop', f'inputs_prior_prediction_iteration_{j}_channel_' + str(i)+'.nii.gz'))
                #     batchdata = list_data_collate(batchdata_list)

                engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                engine.network.eval()

                #Printing which device the image is on prior to the inner loop prediction.
                logger.info(f'The input image prior to the inner loop inference: Image is on cuda: {inputs.is_cuda}')

                with torch.no_grad():
                    logger.info(f'The model prior to the inner loop inference is on device {next(engine.network.parameters()).device}') 
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            #Runs the inferer on the cuda device
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)
                batchdata.update({CommonKeys.PRED: predictions})
                
                ##################################################################################################################
                # #verification check of the prediction generated by the forward pass using Autoseg/Deepgrow inputs. TODO: Delete this.

                # if not self.train:
                #     if not os.path.exists(os.path.join(output_dir, 'TrainingInnerLoopPrediction')):
                #         os.makedirs(os.path.join(output_dir, 'TrainingInnerLoopPrediction'))

                #     for i in range(batchdata[CommonKeys.PRED].size(dim=1)):
                        
                #         placeholder_tensor = batchdata[CommonKeys.PRED][0].cpu()
                #         placeholder = np.array(placeholder_tensor[i], dtype=np.float32)
                #         #print(placeholder)
                #         nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoopPrediction', f'predictions_iteration_{j}_channel_' + str(i)+'.nii.gz'))

                #     ######################################################################################################

                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)
                #Checking whether pred, Image metatensor and label metatensor are on cuda device here:
                logger.info(f'The pre-click transform inputs: Image is on cuda: {batchdata_list[0]["image"].is_cuda}, Label is on cuda {batchdata_list[0]["label"].is_cuda}, Prediction is on cuda {batchdata_list[0]["pred"].is_cuda}')
                for i in range(len(batchdata_list)):
                    batchdata_list[i][self_dict['click_probability_key']] = 1.0
        
                    batchdata_list[i] = self_dict['transforms'](batchdata_list[i])

                    ########################################################################################################
                # if not self.train:

                #     for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                #         placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                #         placeholder = np.array(placeholder_tensor[i])
                #         #print(placeholder)
                #         nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoop', f'inputs_iteration_{j}_channel_' + str(i)+'.nii.gz'))
                #         #################################################################################################################

                batchdata = list_data_collate(batchdata_list)
                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        
        
            ######################## Deepedit validation will occur on the final iteration only.. ###############################
            
            batchdata_val_deepedit = copy.deepcopy(batchdata)
            
            #First we generate the deepedit validation predictions:
            inputs_val_deepedit, _ = engine.prepare_batch(batchdata_val_deepedit)
            inputs_val_deepedit = inputs_val_deepedit.to(engine.state.device)
            
            # engine.network.eval()
            # with torch.no_grad():
            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_deepedit = engine.inferer(inputs_val_deepedit, engine.network)
                else:
                    predictions_val_deepedit = engine.inferer(inputs_val_deepedit, engine.network)
            
            
            
            ################# Process the prediction for metric computation ###########################
            
            deepedit_activation = Activations(softmax=True)(predictions_val_deepedit[0])
            deepedit_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(deepedit_activation)
            

        
            deepedit_dice = DiceHelper(  # type: ignore
                include_background= False,
                sigmoid = False,
                softmax = False, 
                activate = False,
                get_not_nans = False,
                reduction = MetricReduction.MEAN, #MetricReduction.MEAN,
                ignore_empty = True,
                num_classes = None
                )(y_pred=deepedit_discretised_pred.cpu().unsqueeze(dim=0), y=discretised_label.unsqueeze(dim=0))

            ########################### Saving outputs ##########################
            
            

            ################## Saving the metrics #############################
                
            ############# Appending the metric values to csv file ################# 
            fields = [float(deepgrow_dice[0]), float(autoseg_dice[0]), float(deepedit_dice[0])]
            with open(os.path.join(external_val_output_dir, 'validation_scores', 'validation.csv'),'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            ############### Saving the predictions and labels ################################

            # nib.save(nib.Nifti1Image(np.array(deepedit_discretised_pred[0].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepedit_discretised_pred_channel_0.nii.gz'))
            # nib.save(nib.Nifti1Image(np.array(deepedit_discretised_pred[1].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepedit_discretised_pred_channel_1.nii.gz'))
            
        
        
        # first item in batch only
        engine.state.batch = batchdata

        # inputs, _ = engine.prepare_batch(batchdata)

        
        # for i in range(inputs.size(dim=1)):
        #     placeholder_tensor = inputs[0]
        #     placeholder = np.array(placeholder_tensor[i])
        #     #print(placeholder)
        #     nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures','final_inputs_channel_' + str(i)+'.nii.gz'))
        
        # label = batchdata["label"][0]
        # placeholder = np.array(label[0])
        # nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures/label.nii.gz'))

        logger.info(f'For the final inputs the image is on cuda: {batchdata["image"].is_cuda}, the label is on cuda: {batchdata["label"].is_cuda}')
        logger.info(f'For the engine, amp is {engine.amp}')
        return engine._iteration(engine, batchdata)  # type: ignore[arg-type]