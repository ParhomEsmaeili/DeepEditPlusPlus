'''Script which contains the configuration for the networks being used as the backbone for the deepedit++ application (and for performing A.L)'''

'''
Version 0: DynUnet for deepedit++ v1.1
Version 1: UNETR for deepedit++ v1.1
'''

from monai.networks.nets import UNETR, DynUNet

def run_get_network_configs(version_param, self_dict):
    '''Inputs: The dict containing the self attributes from the parent file (deepedit++ configs) and the version param for the network config
       Outputs: A dict containing the networks (this could be just the standard network, but also the network with dropout: this is for basic uncertainty-quantif
       in A.L) 
    '''
    supported_version_params = ['0', '1'] 

    assert version_param in supported_version_params 

    if version_param == '0':
        
        output_dict = dict() 

        output_dict['base_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= 2 * len(self_dict['labels']) + self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        res_block=True,
                    )
        output_dict['dropout_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= 2 * len(self_dict['labels']) + self_dict['number_intensity_ch'], #TODO: change back #2 * len(self.labels) + self.number_intensity_ch,
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        res_block=True,
                        dropout=0.2,
                    )
        return output_dict 
    
    elif version_param == '1':
        
        output_dict = dict() 
    
        output_dict['base_network'] = UNETR(
                        spatial_dims=3,
                        in_channels= 2 * len(self_dict['labels']) + self_dict['number_intensity_ch'], 
                        out_channels=len(self_dict['labels']),
                        img_size=self_dict['spatial_size'],
                        feature_size=64,
                        hidden_size=1536,
                        mlp_dim=3072,
                        num_heads=48,
                        pos_embed="conv",
                        norm_name="instance",
                        res_block=True,
                    )
        

        output_dict['dropout_network'] = UNETR(
                        spatial_dims=3,
                        in_channels= 2 * len(self_dict['labels']) + self_dict['number_intensity_ch'], #TODO: change back # 2 * len(self.labels) + self.number_intensity_ch,
                        out_channels=len(self_dict['labels']),
                        img_size=self_dict['spatial_size'],
                        feature_size=64,
                        hidden_size=1536,
                        mlp_dim=3072,
                        num_heads=48,
                        pos_embed="conv",
                        norm_name="instance",
                        res_block=True,
                        dropout_rate=0.2,
                    )
        return output_dict 