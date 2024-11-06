''' The generation of the transforms_parametrisation_dict which corresponds to the one in the inference setup.py script! (and the inference_config_setup)
    for the pre-processing, and to the train_setup.py script for any click size parametrisation related components.
'''

def run_generate_test_config_dict(args, version_param, infer_click_parametrisations_dict):
    #Args is an argparse object, version param just corresponds to the version described above.

    supported_version_params = ['1']

    # supported_args = ['Spatial Size',
    #                 'Target Spacing', 
    #                 'Divisible Padding Factor', 
    #                 'Sequentiality Mode', 
    #                 'Simulation Click Parametrised Bool',
    #                 'Click Simulation Transforms Version']

    assert version_param in supported_version_params 

    if version_param == '1':

        #In this case, the only parametrisations supported are spatial_size, divisible_padding_factor and target spacing. This corresponds to SIM with padding (like that in the upgrade report)

        config_dict = dict()

        config_dict['Spatial Size'] = args.spatial_size
        assert config_dict['Spatial Size'] != None 

        config_dict['Target Spacing'] = args.target_spacing
        assert config_dict['Target Spacing'] != None 

        config_dict['Divisible Padding Factor'] = args.divisible_padding_factor
        assert config_dict['Divisible Padding Factor'] != None 


        config_dict['Sequentiality Mode'] = args.sequentiality_mode 
        assert config_dict['Sequentiality Mode'] != None
        
        config_dict['Simulation Click Parametrised Bool'] = args.infer_click_parametrised_bool
        assert config_dict['Simulation Click Parametrised Bool'] != None 

        config_dict['Simulation Click Parametrisation'] = infer_click_parametrisations_dict 
        assert type(config_dict['Simulation Click Parametrisation']) == dict

        config_dict['Click Transforms Version Param'] = '1'

    if version_param == '2':

        #In this case, the only parametrisations supported are spatial_size, divisible_padding_factor and target spacing. This corresponds to CIM with padding (like in the upgrade report)

        # config_dict = dict()

        # config_dict['Spatial Size'] = args.spatial_size
        # assert config_dict['Spatial Size'] != None 

        # config_dict['Target Spacing'] = args.target_spacing
        # assert config_dict['Target Spacing'] != None 

        # config_dict['Divisible Padding Factor'] = args.divisible_padding_factor
        # assert config_dict['Divisible Padding Factor'] != None 


        # config_dict['Sequentiality Mode'] = args.sequentiality_mode 
        # assert config_dict['Sequentiality Mode'] != None
        
        # config_dict['Simulation Click Parametrised Bool'] = args.infer_click_parametrised_bool
        # assert config_dict['Simulation Click Parametrised Bool'] != None 

        # config_dict['Simulation Click Parametrisation'] = infer_click_parametrisations_dict 
        # assert type(config_dict['Simulation Click Parametrisation']) == dict

        # config_dict['Click Transforms Version Param'] = '2' 

        raise ValueError("Not yet implemented in the click simulation util")


    return config_dict

