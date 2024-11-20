import json 
import logging 
import os
import nibabel as nib
from os.path import dirname as up
import copy 
import sys
deepeditpp_run_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_run_utils_dir)

deepeditpp_general_utils_dir = os.path.join(up(up(os.path.abspath(__file__))), 'deepeditplusplus_utils')
sys.path.append(deepeditpp_general_utils_dir)

from deepeditplusplus_test_utils.label_saving import label_saving 

logger = logging.getLogger(__name__)

def probabilistic_inner_loop_runner(app, request_templates, device, inference_run_configs, click_simulation_class):
    '''
    Args:
    app: The monailabel app which we are using to perform inference
    request_templates: the basic templates for the inference requests of the three deepedit ++ modes
    device: device that is being used to perform inference

    inference_run_configs: Dict of the inference configs, contains the following keys:

    Infer Run Name: The list which contains the following setup: ["Initialisation Mode"] or [Editing, "Initialisation Mode", "Number of Iters"]

    Image Info: A list containing the image id being performed inference on (i.e. without the extension) and the path to the image. 

    Output Dir: A path denoting the upper level folder for the output directory (which the subfolders containing the segmentations reside in)

    Class Label Config: A dict denoting the class-label to integer code relationship.

    Sequentiality Mode: A str denoting which sequentiality type mode is being used for collecting the click sets, CIM or SIM like.

    Click simulation class, contains the class which can be called in order to generate the clicks as necessary.

    '''

    infer_run_name = inference_run_configs["Infer Run Name"]
    image_info = inference_run_configs["Image Info"]
    output_dir = inference_run_configs["Output Dir"]
    label_configs = inference_run_configs["Class Label Config"]
    sequentiality_mode = inference_run_configs["Sequentiality Mode"]
    click_simulation_class_initialised = click_simulation_class

    debug_click_placement = False 

    assert type(infer_run_name) == list
    assert len(infer_run_name) == 1 or len(infer_run_name) == 3, "The config for the infer run name was invalid"
    assert type(image_info) == list
    assert len(image_info) == 2
    assert type(output_dir) == str
    assert os.path.exists(output_dir)
    assert type(label_configs) == dict

    # assert type(click_simulation_config) == dict 
    
    supported_sequentiality_modes = ['CIM', 'SIM']

    assert sequentiality_mode in supported_sequentiality_modes
    
    image_id = image_info[0]
    image_path = image_info[1]

    if len(infer_run_name) > 1:
        initialisation_mode = infer_run_name[1].title()
        subsequent_mode = infer_run_name[0].title()
        num_iterations = int(infer_run_name[2])

        #Extracting the appropriate request formats:
        initial_request = request_templates[initialisation_mode + "_template"]
        subsequent_request = request_templates[subsequent_mode + "_template"]
        
        #Appending the key:val pair for the image id.
        initial_request["image"] = image_id
        subsequent_request["image"] = image_id 

        #Appending the device name:
        initial_request["device"] = device
        subsequent_request["device"] = device

        #Extracting the path for the GT label that we need to simulate the clicks with (if necessary):
        gt_path = os.path.join(output_dir, 'labels', 'original', image_id + '.nii.gz')

        #Creating a save folder for storing the guidance points used during inference.
        
        guidance_points_save_folder = os.path.join(up(up(gt_path)), f'guidance_points')
        os.makedirs(guidance_points_save_folder, exist_ok=True) 



        if initialisation_mode.title() == "Autoseg":
            res = app.infer(request = initial_request)

            if sequentiality_mode == 'CIM':
                tracked_guidance = dict()
                for label_name in label_configs.keys(): 
                    tracked_guidance[label_name] = [] 
        
        elif initialisation_mode.title() == "Interactive":
            if sequentiality_mode == 'CIM':

                initial_request, transform_output_dict, tracked_guidance = click_simulation_class_initialised(initial_request, clicking_mode = 'interactive', gt_path=gt_path, tracked_guidance=None)
                res = app.infer(request = initial_request)

            elif sequentiality_mode == 'SIM':
                initial_request, transform_output_dict = click_simulation_class_initialised(initial_request, clicking_mode = 'interactive', gt_path=gt_path, tracked_guidance=None)
                res = app.infer(request = initial_request)
            
            #Saving the guidance points (RAS orientation). 
            guidance_points_save = dict()
            for label_name in label_configs.keys():
                guidance_points_save[label_name] = initial_request[label_name]
            

            try:
                with open(os.path.join(guidance_points_save_folder, 'interactive.json'), 'r') as f:
                    saved_dict = json.load(f)
                    saved_dict[image_id] = guidance_points_save 

                with open(os.path.join(guidance_points_save_folder, 'interactive.json'), 'w') as f:
                    json.dump(saved_dict, f)
            except:
                with open(os.path.join(guidance_points_save_folder, 'interactive.json'), 'w') as f:
                    # saved_dict = json.load(f)
                    saved_dict = dict()
                    saved_dict[image_id] = guidance_points_save 
                    json.dump(saved_dict, f)


        # Saving the initialisation segmentation.

        res_savepath = label_saving(res, os.path.join(output_dir, "labels", initialisation_mode.lower()), image_id, image_path)
        

        for i in range(1, num_iterations + 1):
            
            subsequent_request["previous_seg"] = res_savepath 

            # The function which will simulate click points given the existing segmentation, and the ground truth. 
            if sequentiality_mode == "SIM":
                subsequent_request, transform_output_dict = click_simulation_class_initialised.__call__(subsequent_request, clicking_mode = 'deepedit', gt_path = gt_path, tracked_guidance=None)
                #Saving the guidance points (RAS orientation). 

            
                guidance_points_save = dict()
                for label_name in label_configs.keys():
                    guidance_points_save[label_name] = subsequent_request[label_name]

                if i != num_iterations:
                    try:
                        with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'r') as f:
                            saved_dict = json.load(f)
                            saved_dict[image_id] = guidance_points_save
                        with open(os.path.join(guidance_points_save_folder, f'deepedit_iteration_{i}.json'), 'w') as f:
                            json.dump(saved_dict, f) 
                    except:
                        with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'w') as f:
                            # saved_dict = json.load(f)
                            saved_dict = dict()
                            saved_dict[image_id] = guidance_points_save
                            json.dump(saved_dict, f) 
                else:
                    try:
                        with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'r') as f:
                            saved_dict = json.load(f)
                            saved_dict[image_id] = guidance_points_save
                        with open(os.path.join(guidance_points_save_folder, f'final_iteration.json'), 'w') as f:
                            json.dump(saved_dict, f) 
                    except:
                        with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'w') as f:
                            # saved_dict = json.load(f)
                            saved_dict = dict()
                            saved_dict[image_id] = guidance_points_save
                            json.dump(saved_dict, f) 

            elif sequentiality_mode == "CIM":
                tracked_guidance_input = copy.deepcopy(tracked_guidance)
                subsequent_request, transform_output_dict, tracked_guidance = click_simulation_class_initialised.__call__(subsequent_request, clicking_mode = 'deepedit', gt_path = gt_path, tracked_guidance=None)      

                #Saving the guidance points (RAS orientation) with the tracked guidance. 

            
                guidance_points_save = dict()
                for label_name in label_configs.keys():
                    guidance_points_save[label_name] = tracked_guidance_input[label_name]

                if i != num_iterations:
                    try:
                        with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'r') as f:
                            saved_dict = json.load(f)
                            saved_dict[image_id] = guidance_points_save
                        with open(os.path.join(guidance_points_save_folder, f'deepedit_iteration_{i}.json'), 'w') as f:
                            json.dump(saved_dict, f) 
                    except:
                        with open(os.path.join(guidance_points_save_folder,f'deepedit_iteration_{i}.json'), 'w') as f:
                            # saved_dict = json.load(f)
                            saved_dict = dict()
                            saved_dict[image_id] = guidance_points_save
                            json.dump(saved_dict, f) 
                else:
                    try:
                        with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'r') as f:
                            saved_dict = json.load(f)
                            saved_dict[image_id] = guidance_points_save
                        with open(os.path.join(guidance_points_save_folder, f'final_iteration.json'), 'w') as f:
                            json.dump(saved_dict, f) 
                    except:
                        with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'w') as f:
                            # saved_dict = json.load(f)
                            saved_dict = dict()
                            saved_dict[image_id] = guidance_points_save
                            json.dump(saved_dict, f) 

            #Transform output dict just contains the dictionary of data which has been passed through the transforms (e.g. the labels!) 

            if debug_click_placement:
                #If we want to output files to do a sanity check:
                debug_click_placement_func() 
            
            if sequentiality_mode == 'SIM':

                # Running editing iter inference with the prev_label and the corrective clicks
                res = app.infer(request = subsequent_request)
            
            elif sequentiality_mode == 'CIM':

                #We delete the "previous seg" because the CIM should not have any of this information during editing.

                del subsequent_request["previous_seg"]

                # Running editing iter inference with the full set of clicks.
                res = app.infer(request = subsequent_request)

            #For each iteration we have a saved label.

            #We denote the save path for the next iteration of editing simulation in order to call "previous seg"
            if i != num_iterations:
                res_savepath = label_saving(res, os.path.join(output_dir, "labels", 'deepedit_iteration_' + str(i)), image_id, image_path)
            else:
                _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)
        
    else:
        input_request = request_templates[infer_run_name[0] + "_template"]

        #Adding the image id to the input request
        input_request["image"] = image_id
        
        #Task name:
        print(infer_run_name[0])
        initialisation = infer_run_name[0]
        
        if initialisation.title() == 'Autoseg':

            res = app.infer(request=input_request)
            
            _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)

            tracked_guidance = dict()
            for key in label_configs.keys(): 
                tracked_guidance[key] = []

        
        elif initialisation.title() == "Interactive":
            #Extracting the path for the GT label that we need to simulate the clicks with:
            gt_path = os.path.join(output_dir, 'labels', 'original', image_id + '.nii.gz')
            
            if sequentiality_mode == 'SIM':
                input_request, transform_output_dict = click_simulation_class_initialised.__call__(input_request, clicking_mode = initialisation.lower(), gt_path = gt_path, tracked_guidance=None)
            elif sequentiality_mode == 'CIM':
                input_request, transform_output_dict, tracked_guidance = click_simulation_initialised.__call__(input_request,  clicking_mode = initialisation.lower(), gt_path = gt_path, tracked_guidance=None)
            
            res = app.infer(request=input_request)
            
            _ = label_saving(res, os.path.join(output_dir, "labels", "final"), image_id, image_path)


            #Creating a save folder for storing the guidance points.
        
            guidance_points_save_folder = os.path.join(up(up(gt_path)), f'guidance_points')
            os.makedirs(guidance_points_save_folder, exist_ok=True) 

            #Saving the guidance points (RAS orientation)
            guidance_points_save = dict()
            for label_name in label_configs.keys():
                guidance_points_save[label_name] = input_request[label_name]


            try:
                with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'r') as f:
                    saved_dict = json.load(f)
                    saved_dict[image_id] = guidance_points_save
                with open(os.path.join(guidance_points_save_folder, f'final_iteration.json'), 'w') as f:
                    json.dump(saved_dict, f) 
            except:
                with open(os.path.join(guidance_points_save_folder,f'final_iteration.json'), 'w') as f:
                    # saved_dict = json.load(f)
                    saved_dict = dict()
                    saved_dict[image_id] = guidance_points_save
                    json.dump(saved_dict, f) 


    logger.info(f'Inference completed for image: {image_id}') 


#TODO: 
'''
For dynamic click size and fixed click size (with a parametrisation), (with click size as an additional input field) this has not yet been implemented.
'''


def debug_click_placement_func(): #i, num_iterations, output_dir, subsequent_task, image_id, transforms_output_dict):

    pass 
    ################## Saving the discrepancies at iteration i of the editing ###########################
            #Discrepancy save folder:



    # if i != num_iterations:
    #     discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' padded_discrepancy', 'iteration_' + str(i), image_id)
    #     os.makedirs(discrepancy_folder_path, exist_ok=True)

    #     for label_class in discrepancy_output_dict["label_names"].keys():
            
    #         nib.save(nib.Nifti1Image(np.array(discrepancy_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
    # else:
    #     discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' padded_discrepancy', 'final', image_id)
    #     os.makedirs(discrepancy_folder_path, exist_ok=True)

    #     for label_class in discrepancy_output_dict["label_names"].keys():
            
    #         nib.save(nib.Nifti1Image(np.array(discrepancy_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
    
    # #Saving the full sized discrepancy images in order to examine the validity of the guidance points generated.
    
    # if i != num_iterations:
    #     discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' original_discrepancy', 'iteration_' + str(i), image_id)
    #     os.makedirs(discrepancy_folder_path, exist_ok=True)

    #     for label_class in transform_output_dict["label_names"].keys():
            
    #         nib.save(nib.Nifti1Image(np.array(transform_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
    # else:
    #     discrepancy_folder_path = os.path.join(output_dir, 'labels', subsequent_task + ' original_discrepancy', 'final', image_id)
    #     os.makedirs(discrepancy_folder_path, exist_ok=True)

    #     for label_class in transform_output_dict["label_names"].keys():
            
    #         nib.save(nib.Nifti1Image(np.array(transform_output_dict[f"discrepancy_{label_class}"])[0], None), os.path.join(discrepancy_folder_path, f"discrepancy_{label_class}.nii.gz"))
    # ######################################################################################################
