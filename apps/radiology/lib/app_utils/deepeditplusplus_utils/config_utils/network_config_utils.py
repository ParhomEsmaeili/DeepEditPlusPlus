'''Script which contains the configuration for the networks being used as the backbone for the deepedit++ application (and for performing A.L)'''

'''
Version -5: Error detected, modified version -1 so that it is properly configured in a similar capacity to the fullres 3d nnU-Net RE: the quantity of deep supervision loss terms.
We were previously using 1 too many.

Version -4: Dynunet configured in the same manner as version -1, but with dropout = 0.2

Version -3: DynUnet for a basic U-net which is configured in a similar capacity to a fullres 3D nnU-net WITH deep supervision AND residual connections. 

Version -2: DynUnet for a basic U-net (i.e. no extra channels for clicks only the image) which is otherwise configured in the same manner as version 0.
Version -1: DynUnet for a basic U-net (i.e. no extra channels only the image) which is mostly configured in the same capacity as fullres 3D nnU-Net,
the only difference is that we have affine=False for instance norm and bias = False for the convolutional layers, in line with the fact that nnU-Net uses bias
on BOTH (which would cancel out.).

Version 0: DynUnet for deepedit++ v1.1
Version 1: UNETR for deepedit++ v1.1

'''

from monai.networks.nets import UNETR, DynUNet
import os
import sys
from os.path import dirname as up

# sys.path.append(os.path.abspath(up(up(up(up(up(__file__)))))))

# from networks.nets import UNETR, DynUNet 

def run_get_network_configs(version_param, self_dict):
    '''Inputs: The dict containing the self attributes from the parent file (deepedit++ configs) and the version param for the network config
       Outputs: A dict containing the networks (this could be just the standard network, but also the network with dropout: this is for basic uncertainty-quantif
       in A.L or even just for regularisation) 
    '''
    supported_version_params = ['-5', '-4', '-3', '-2','-1', '0', '1'] 

    assert version_param in supported_version_params 

    if version_param == '-5':
        
        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=True,
                        deep_supr_num = 3, #
                        res_block=False,
                    )
        output_dict['infer_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=False,
                        deep_supr_num = 3, #
                        res_block=False,
                    )

        output_dict['epistemic_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=False,
                        deep_supr_num=3,
                        res_block=False,
                        dropout=0.2,
                    )
        return output_dict 

    elif version_param == '-4':
        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=True,
                        deep_supr_num = 4,
                        res_block=False,
                        dropout=0.2,
                    )
        
        output_dict['infer_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        deep_supr_num = 4,
                        res_block=False
                    )

        output_dict['epistemic_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'], #TODO: change back #2 * len(self.labels) + self.number_intensity_ch,
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        deep_supr_num = 4,
                        res_block=False,
                        dropout=0.2,
                    )
        return output_dict


    elif version_param == '-3':
        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=True,
                        deep_supr_num = 4,
                        res_block=True,
                    )
        
        output_dict['infer_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=True,
                        deep_supr_num = 4,
                        res_block=True,
                    )

        output_dict['epistemic_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'], #TODO: change back #2 * len(self.labels) + self.number_intensity_ch,
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=True,
                        deep_supr_num = 4,
                        res_block=True,
                        dropout=0.2,
                    )
        return output_dict

    if version_param == '-2':

        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        res_block=True,
                    )

        output_dict['infer_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance",
                        deep_supervision=False,
                        res_block=True,
                    )

        output_dict['epistemic_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels=self_dict['number_intensity_ch'], #TODO: change back #2 * len(self.labels) + self.number_intensity_ch,
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

    elif version_param == '-1':
        
        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=True,
                        deep_supr_num = 4, #
                        res_block=False,
                    )
        output_dict['infer_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=False,
                        deep_supr_num = 4, #
                        res_block=False,
                    )

        output_dict['epistemic_network'] = DynUNet(
                        spatial_dims=3,
                        in_channels= self_dict['number_intensity_ch'],
                        out_channels=len(self_dict['labels']),
                        kernel_size=[3, 3, 3, 3, 3, 3],
                        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                        norm_name="instance", #("instance", {'affine':True}),
                        deep_supervision=False,
                        deep_supr_num=4,
                        res_block=False,
                        dropout=0.2,
                    )
        return output_dict 

    elif version_param == '0':
        
        output_dict = dict() 

        output_dict['train_network'] = DynUNet(
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

        output_dict['infer_network'] = DynUNet(
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

        output_dict['epistemic_network'] = DynUNet(
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
    
        output_dict['train_network'] = UNETR(
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
        
        output_dict['infer_network'] = UNETR(
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

        output_dict['epistemic_network'] = UNETR(
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