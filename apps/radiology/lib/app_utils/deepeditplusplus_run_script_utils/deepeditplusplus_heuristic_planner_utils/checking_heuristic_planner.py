from monailabel.utils.others.generic import get_bundle_models, strtobool
import os
from os.path import dirname as up    
import json

'''Class which performs whichs as to whether a heuristic planner is required or not (i.e. whether the bool should be on or not), e.g. depending on the modality of the dataset, 
    or the support for configuring the network, or the resampling, etc. 
'''

'''
    Version 1: Checks whether the modality is x-ray based or not. If not, then we use per image normalisation.
'''
class HeuristicPlannerChecker:

    def __init__(self,
                args: dict):


        supported_version_params = ['1']
        supported_heuristic_planner_bools = ['True', 'False']
        
        if args['heuristic_planner_version'] not in supported_version_params:
            raise ValueError('The heuristic planner version must match the version of the checker.') 
        if args['heuristic_planner'].title() not in supported_heuristic_planner_bools:
            raise ValueError('Invalid type for the heuristic planner bool.')

        self.version_param = args['heuristic_planner_version']

        self.args_dict = args

    def modality_checker(self):

        x_ray_modalities = ['X-Ray', 'CT']
        
        ###################### Setting the location to extract the dataset configs from ####################
        
        codebase_dir_name = up(up(up(up(up(up(up(__file__)))))))
        
        dataset_json_path = os.path.join(codebase_dir_name, 'datasets', self.args_dict['studies'], 'dataset.json')

        with open(dataset_json_path) as f:
            dataset_config_dict = json.load(f)

            
            if dataset_config_dict['modality'] in x_ray_modalities:
                if self.args['heuristic_planner']  == 'False':
                    raise ValueError('The heuristic planner cannot be false if we are using an X-ray type modality')


    def __call__(self):

        if self.version_param == '1':
            self.modality_checker()
        
