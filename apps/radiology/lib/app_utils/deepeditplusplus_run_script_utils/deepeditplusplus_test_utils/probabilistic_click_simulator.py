import json 
import os
import nibabel as nib
from os.path import dirname as up
import sys
deepeditpp_run_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_run_utils_dir)

# deepeditpp_general_utils_dir = os.path.join(up(up(os.path.abspath(__file__))), 'deepeditplusplus_utils')
# sys.path.append(deepeditpp_general_utils_dir)


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ToNumpyd,
    Resized,
    DivisiblePadd
) 
# from monailabel.deepeditPlusPlus.transforms import (
#     MappingLabelsInDatasetd,
#     NormalizeLabelsInDatasetd,
#     FindAllValidSlicesMissingLabelsd,
#     AddInitialSeedPointMissingLabelsd,
#     FindDiscrepancyRegionsDeepEditd,
#     AddRandomGuidanceDeepEditd
# )

from deepeditplusplus_test_transforms.add_initial_seed_point_missing_labelsd import AddInitialSeedPointMissingLabelsd
from deepeditplusplus_test_transforms.find_all_valid_slices_missinglabelsd import FindAllValidSlicesMissingLabelsd
from deepeditplusplus_test_transforms.normalize_labels_in_datasetd import NormalizeLabelsInDatasetd 
from deepeditplusplus_test_transforms.find_discrepancy_regionsd import FindDiscrepancyRegionsDeepEditd
from deepeditplusplus_test_transforms.cim_add_random_guidance_test_simd import AddRandomTestGuidanceCIMDeepEditd
from deepeditplusplus_test_transforms.sim_add_random_guidance_test_simd import AddRandomTestGuidanceSIMDeepEditd



'''Parametrised click simulation class, which calls on transforms which are also parametrised. '''


class probabilistic_click_simulation:

    def __init__(self, 
                class_label_configs: dict,  
                click_sim_config_dict: dict):

        self.class_label_configs = class_label_configs
        #The click sim configuration dict contains all of the information about the parametrisations necessary for the click simulation transforms:

        #It contains the version param for the list of transforms being performed/i.e. what version of the click simulation func will be used.
        #It also contains the parametrisations for the corresponding transforms in that version param.



        supported_sequentiality_modes = ['SIM', 'CIM']
        supported_click_parametrised_bool = [False]
        supported_click_parametrisation_types = ['No Click Param']
        
        self.click_sim_config_dict = click_sim_config_dict 

        assert self.click_sim_config_dict['Sequentiality Mode'] in supported_sequentiality_modes
        assert self.click_sim_config_dict['Simulation Click Parametrised Bool'] in supported_click_parametrised_bool 
        assert type(self.click_sim_config_dict['Simulation Click Parametrisation']) == dict
        self.click_parametrisation_type = list(self.click_sim_config_dict['Simulation Click Parametrisation'].keys())[0]
        assert self.click_parametrisation_type in supported_click_parametrisation_types 

        self.sequentiality_mode = self.click_sim_config_dict
        self.label_configs = class_label_configs

    def click_simulation_sim_version_param_1(self, inference_request, clicking_mode, gt_path):
        '''
        Inference request is the dict containing the base structure for the deepedit/interactive init inference request, without the addition of the 
        click points/guidance points.

        Clicking mode: The clicking mode

        GT_path : The full path to the ground truth label.
        '''
        


        input_dict = dict()

        input_dict["class_labels"] = self.label_configs
    
        #GT label path
        input_dict["label"] = gt_path

        assert os.path.exists(gt_path), "The ground truth did not exist" 


        if clicking_mode == "interactive": #Interactive initialisation

            composed_transform = [
            LoadImaged(keys=("label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=input_dict["class_labels"], version_param='0'), 
            #We must orientate to RAS so that the guidance points are in the correct coordinate system for the inference script.
            Orientationd(keys=["label"], axcodes="RAS"),
            # Transforms for click simulation (depracated)
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids", version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids", version_param='0')
            ]

        elif clicking_mode == "deepedit":

            #Adding the prev_seg path:
            input_dict["previous_seg"] = inference_request["previous_seg"]

            assert os.path.exists(input_dict["previous_seg"])
        
            composed_transform = [
            LoadImaged(keys=("previous_seg", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("previous_seg", "label")),
            # We normalise so that it is in the expected format (?)
            NormalizeLabelsInDatasetd(keys=("previous_seg", "label"), label_names= input_dict["class_labels"], version_param='0'), 
            #We need to re-orientate ourselves in RAS so that we generate guidance points in RAS coordinates:
            Orientationd(keys=("previous_seg", "label"), axcodes="RAS"),
            ToNumpyd(keys=("previous_seg", "label")),
            # Transforms for click simulation
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="previous_seg", discrepancy="discrepancy", version_param='0'),
            AddRandomTestGuidanceSIMDeepEditd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                version_param='1'
            ),
            ]
        
        transform_output_dict = Compose(transforms=composed_transform, map_items = False)(input_dict)
       
     
        if clicking_mode == "deepedit": 
            
            #Converting the guidance clicks to inputs for the inference script
            for key in transform_output_dict["guidance"].keys():
                sim_clicks = transform_output_dict["guidance"][key]
                sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
                inference_request[key] = sim_click_valid
        #    inference_request["guidance"] = transform_output_dict["guidance"]
                
            return inference_request, transform_output_dict

        elif clicking_mode == "interactive":
            #Converting the guidance clicks to inputs for the inference script
            for key in transform_output_dict["guidance"].keys():
                sim_clicks = ast.literal_eval(transform_output_dict["guidance"][key])
                sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
                inference_request[key] = sim_click_valid
        #    inference_request["guidance"] = transform_output_dict["guidance"]
            return inference_request, transform_output_dict

    def click_simulation_cim_version_param_1(self, inference_request, clicking_mode, gt_path, tracked_guidance):
        '''

        inference request is the dict containing the base structure for the deepedit/deepgrow inference request, without the addition of the 
        click points/guidance points.

        guidance is the previously provided guidance (IF IT HAS ONE! otherwise its just a dict with empty lists for each class)
        '''
        input_dict = dict()

        input_dict["class_labels"] = self.label_configs
        
        #GT label path
        input_dict["label"] = gt_path

        #I.e. if we are editing we extract the tracked guidance.
        if clicking_task == "deepedit":
            input_dict["guidance"] = tracked_guidance 

        if clicking_task == "interactive":
            composed_transform = [
            LoadImaged(keys=("label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=input_dict["class_labels"], version_param='0'), 
            #We must orientate to RAS so that the guidance points are in the correct coordinate system for the inference script.
            Orientationd(keys=["label"], axcodes="RAS"),
            # Transforms for click simulation (depracated)
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids", version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids", version_param='0')
            ]
        elif clicking_task == "deepedit":
            #Adding the prev_seg path:
            input_dict["previous_seg"] = inference_request["previous_seg"]

            composed_transform = [
            LoadImaged(keys=("previous_seg", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("previous_seg", "label")),
            # We normalise so that it is in the expected format (?)
            NormalizeLabelsInDatasetd(keys=("previous_seg", "label"), label_names= input_dict["class_labels"], version_param='0'), 
            #We need to re-orientate ourselves in RAS so that we generate guidance points in RAS coordinates:
            Orientationd(keys=("previous_seg", "label"), axcodes="RAS"),
            ToNumpyd(keys=("previous_seg", "label")),
                # Transforms for click simulation
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="previous_seg", discrepancy="discrepancy", version_param='0'),
            AddRandomTestGuidanceCIMDeepEditd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                version_param='1'
            ),
            ]
        
        transform_output_dict = Compose(transforms=composed_transform, map_items = False)(input_dict)
        
        ################### Above generates guidance points which are denoted under the output_dict with the guidance key ######################
        #         
        if clicking_task == "deepedit": 

            
            #Save a final guidance separately.
             
            final_guidance = dict()

            #Converting the guidance clicks to inputs for the inference script
            for key in transform_output_dict["guidance"].keys():
                sim_clicks = transform_output_dict["guidance"][key]
                sim_click_valid = []
                for point in sim_clicks:
                    if len(point) == 3:
                        sim_click_valid.append(point)
                    else:
                        if point[0] >=1:
                            sim_click_valid.append(point[1:])

                #sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
                inference_request[key] = sim_click_valid
                final_guidance[key] = sim_click_valid

        #    inference_request["guidance"] = transform_output_dict["guidance"]
            

            return inference_request, transform_output_dict, final_guidance

        elif clicking_task == "interactive":
            #Save a final guidance separately since this original version of deepedit requires it!
            final_guidance = dict()

            #Converting the guidance clicks to inputs for the inference script
            for key in transform_output_dict["guidance"].keys():
                sim_clicks = ast.literal_eval(transform_output_dict["guidance"][key])
                sim_click_valid = [click[1:] for click in sim_clicks if click[0] >= 1]
                inference_request[key] = sim_click_valid
                final_guidance[key] = sim_click_valid 
        #    inference_request["guidance"] = transform_output_dict["guidance"]
            return inference_request, transform_output_dict, final_guidance


    def __call__(self, base_infer_request_dict, clicking_mode, gt_path, tracked_guidance):

        if self.click_parametrisation_type == 'No Click Param':

            if self.click_sim_config_dict['Click Transforms Version Param'] == '1':

                if self.click_sim_config_dict['Sequentiality Mode'] == "SIM":

                    return self.click_simulation_sim_version_param_1(base_infer_request_dict, clicking_mode, gt_path)

                elif self.click_sim_config_dict['Sequentiality Mode'] == "CIM":

                    return self.click_simulation_cim_version_param_1(base_infer_request_dict, clicking_mode, gt_path, tracked_guidance)
        

        elif self.click_parametrisation_type == 'Dynamic Click Size':

            raise ValueError("This click size parametrisation type is not yet supported")

        elif self.click_parametrisation_type == 'Fixed Click Size':

            raise ValueError("This click size parametrisation type is not yet supported")     
