'''
This script parametrises all of the information used in the definition of the train_setup.py for deepeditplusplus. 
'''

import logging

import torch
import json 

# from monai.handlers import MeanDice, from_engine
from monai.inferers import SimpleInferer
# from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    SelectItemsd,
    ToNumpyd,
    ToTensord,
    ToDeviced,
    DivisiblePadd,
    CenterSpatialCropd
)

# from monailabel.deepeditPlusPlus.handlers import TensorBoardImageHandler
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

############################

#Imports from utils folders

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)



from train_utils.get_click_transform_setup import run_get_click_transf
from train_utils.loss_function_setup import run_get_loss_func
from train_utils.train_pre_transf_setup import run_get_train_pre_transf
from train_utils.val_pre_transf_setup import run_get_val_pre_transf

from transforms_utils.splitpredlabelsd import SplitPredsLabeld
from transforms_utils.extract_mapsd import ExtractMapsd

from inner_loop_utils.inner_loop_setup import Interaction

app_dir = up(up(up(up(os.path.abspath(__file__)))))

from monai_handlers import (
    LrScheduleHandler,
    MeanDice,
    from_engine
)

logger = logging.getLogger(__name__)


class DeepEditPlusPlus(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        labels,
        external_validation_dir,
        n_saved,
        modality,
        train_version_params,
        component_parametrisation_dict,
        #extract_channels,
        description="Train DeepEdit++ model for 3D Images",
        number_intensity_ch=1,
        debug_mode=False,
        **kwargs,
    ):
        self._network = network
        self.modality = modality
        self.external_validation_dir = external_validation_dir

        #Extracting the fixed variables. 
        
        self.number_intensity_ch = number_intensity_ch
        
        self.debug_mode = debug_mode
        
        #Extracting the parametrisations required for the compose components (e.g. spatial size, padding size etc, probability for the inner loop etc.)

        #The dict for this can be found in the train config setup file. 
        self.component_parametrisation_dict = component_parametrisation_dict


        #Extracting the version params for the components.
        self.version_params = train_version_params 
        #####################################################################

        self.optimizer_version_param = self.version_params["optimizer_version_param"]
        self.lr_scheduler_version_param = self.version_params["lr_scheduler_version_param"] 
        self.loss_func_version_param = self.version_params["loss_func_version_param"]
        self.get_click_version_param = self.version_params["get_click_version_param"] 
        self.train_pre_transforms_version_param = self.version_params["train_pre_transforms_version_param"] 
        self.train_post_transforms_version_param = self.version_params["train_post_transforms_version_param"] 
        self.val_pre_transforms_version_param = self.version_params["val_pre_transforms_version_param"]
        self.val_post_transforms_version_param = self.version_params["val_post_transforms_version_param"]
        self.train_inferer_version_param = self.version_params["train_inferer_version_param"] 
        self.val_inferer_version_param = self.version_params["val_inferer_version_param"]
        self.train_iter_update_version_param = self.version_params["train_iter_update_version_param"] 
        self.val_iter_update_version_param = self.version_params["val_iter_update_version_param"] 
        self.train_key_metric_version_param = self.version_params["train_key_metric_version_param"]
        self.val_key_metric_version_param = self.version_params["val_key_metric_version_param"] 
        self.train_handler_version_param = self.version_params["train_handler_version_param"] 
        self.engine_version_param = self.version_params["engine_version_param"]

        self.supported_optimizer  = ['0']
        self.supported_lr_scheduler = ['0']
        self.supported_loss_func = ['-1', '0', '1', '2', '3', '4']
        self.supported_get_click_transform = ['1', '2']
        self.supported_train_pre_transf = ['-6','-5', '-4', '-3','-2', '-1','1','2', '3']
        self.supported_train_post_transf = ['1', '2']
        self.supported_val_pre_transf = ['-3','-2', '-1', '1','2', '3']
        self.supported_val_post_transf = ['1']
        self.supported_train_inferer = ['0']
        self.supported_val_inferer = ['0']
        self.supported_train_iter = ['-1','1','2', '3', '4']
        self.supported_val_iter = ['-1', '1','2', '3', '4']
        self.supported_train_metric = ['1']
        self.supported_val_metric = ['1']
        self.supported_train_handlers = ['0'] 
        self.supported_engine_versions = ['0', '1']

        assert self.optimizer_version_param in self.supported_optimizer 
        assert self.lr_scheduler_version_param in self.supported_lr_scheduler 
        assert self.loss_func_version_param in self.supported_loss_func 
        assert self.get_click_version_param in self.supported_get_click_transform  
        assert self.train_pre_transforms_version_param in self.supported_train_pre_transf  
        assert self.train_post_transforms_version_param in self.supported_train_post_transf 
        assert self.val_pre_transforms_version_param in self.supported_val_pre_transf
        assert self.val_post_transforms_version_param in self.supported_val_post_transf
        assert self.train_inferer_version_param in self.supported_train_inferer
        assert self.val_inferer_version_param in self.supported_val_inferer
        assert self.train_iter_update_version_param in self.supported_train_iter 
        assert self.val_iter_update_version_param in self.supported_val_iter  
        assert self.train_key_metric_version_param in self.supported_train_metric 
        assert self.val_key_metric_version_param in self.supported_val_metric  
        assert self.train_handler_version_param in self.supported_train_handlers  
        assert self.engine_version_param in self.supported_engine_versions

        
    
        super().__init__(model_dir, description, n_saved=n_saved, labels=labels, engine_version_param=self.engine_version_param, **kwargs)
        
        # self.self_dict = dict(vars(self))

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        if self.optimizer_version_param == '0':

            return torch.optim.Adam(context.network.parameters(), lr=0.0001)

    def lr_scheduler_handler(self, context: Context):
        if self.lr_scheduler_version_param == '0':
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(context.optimizer, mode="min")
            # return LrScheduleHandler(lr_scheduler, print_lr=True, step_transform=lambda x: x.state.output[0]["loss"])

            lr_scheduler = torch.optim.lr_scheduler.StepLR(context.optimizer, step_size=1000, gamma=0.1)
            return LrScheduleHandler(lr_scheduler, print_lr=True)

    def loss_function(self, context: Context):
        
        return run_get_loss_func(dict(vars(self)), context, self.loss_func_version_param)

    def get_click_transforms(self, context: Context):
        
        return run_get_click_transf(dict(vars(self)), context, self.get_click_version_param)

    def train_pre_transforms(self, context: Context):
        
        return run_get_train_pre_transf(dict(vars(self)), context, self.train_pre_transforms_version_param)

        
    def train_post_transforms(self, context: Context):
        
        if self.train_post_transforms_version_param == '1':
            #This is only configured for single output map formulations, not for multi-feature maps generated by deep supervision.
            return [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=len(self._labels),
                ),
                SplitPredsLabeld(keys="pred", version_param='0'),#,version_param='0'),
            ]
        elif self.train_post_transforms_version_param == '2':
            #This is only configured for single output map formulations, not for deep supervision.
            return [
                ExtractMapsd(keys="pred", version_param='0'),
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=len(self._labels),
                ),
                SplitPredsLabeld(keys="pred", version_param='0')
                # Activationsd(keys="pred_output", softmax=True),
                # AsDiscreted(
                #     keys=("pred_output", "label"),
                #     argmax=(True, False),
                #     to_onehot=len(self._labels),
                # ),
                # SplitPredsLabeld(keys="pred_output", version_param='1'),#,version_param='0'),
            ]

    def val_post_transforms(self, context: Context):
        
        if self.val_post_transforms_version_param == '1':
            #This is only configured for single output map formulations, not for multi-feature maps. Therefore we keep it fully separate from train post transf
            #which may require handling for deep supervision heads.
            return [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=len(self._labels),
                ),
                SplitPredsLabeld(keys="pred", version_param='0'),#,version_param='0'),
            ]
        

    def val_pre_transforms(self, context: Context):
        
        return run_get_val_pre_transf(dict(vars(self)), context, self.val_pre_transforms_version_param)

    def train_inferer(self, context: Context):
        if self.train_inferer_version_param == '0':
            return SimpleInferer()

    def val_inferer(self, context: Context):
        if self.val_inferer_version_param == '0':
            return SimpleInferer()

    def train_iteration_update(self, context: Context):
        
        if self.train_iter_update_version_param == '-1':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=None,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=None,
                click_probability_key=None,
                train=True,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )

        elif self.train_iter_update_version_param == '1':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=True,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.train_iter_update_version_param == '2':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=True,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.train_iter_update_version_param == '3':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=True,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.train_iter_update_version_param == '4':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=True,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )

    def val_iteration_update(self, context: Context):
        
        if self.val_iter_update_version_param == '-1':

            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=None,
                num_intensity_channel=self.number_intensity_ch,  
                transforms=None,
                click_probability_key=None,
                train=False,
                version_param=self.train_iter_update_version_param 
                #label_names=self._labels,
            )

        elif self.val_iter_update_version_param == '1':
        
            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch, 
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=False,
                version_param=self.val_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.val_iter_update_version_param == '2':
        
            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch, 
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=False,
                version_param=self.val_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.val_iter_update_version_param == '3':
        
            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch, 
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=False,
                version_param=self.val_iter_update_version_param 
                #label_names=self._labels,
            )
        
        elif self.val_iter_update_version_param == '4':
        
            return Interaction(
                self_dict=dict(vars(self)),
                external_validation_output_dir=self.external_validation_dir,
                num_intensity_channel=self.number_intensity_ch, 
                transforms=self.get_click_transforms(context),
                click_probability_key="probability",
                train=False,
                version_param=self.val_iter_update_version_param 
                #label_names=self._labels,
            )


    def train_key_metric(self, context: Context):

        if self.train_key_metric_version_param == '1':
            #This is only configured for single output map formulations, not for deep supervision.
            all_metrics = dict()
            all_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]), include_background=False)
            for key_label in self._labels:
                if key_label != "background":
                    all_metrics[key_label + "_dice"] = MeanDice(
                        output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                    )
            return all_metrics

    def val_key_metric(self, context: Context):
        
        if self.val_key_metric_version_param == '1':
            #This is only configured for single output map formulations, not for deep supervision.
            all_metrics = dict()
            all_metrics["val_mean_dice"] = MeanDice(
                output_transform=from_engine(["pred", "label"]), include_background=False
            )
            for key_label in self._labels:
                if key_label != "background":
                    all_metrics["val_" + key_label + "_dice"] = MeanDice(
                        output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                    )
            return all_metrics

    def train_handlers(self, context: Context):
        
        if self.train_handler_version_param == '0':
            #This is only configured for single output map formulations, not for deep supervision.
            handlers = super().train_handlers(context)
            if self.debug_mode and context.local_rank == 0:
                handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
            return handlers
