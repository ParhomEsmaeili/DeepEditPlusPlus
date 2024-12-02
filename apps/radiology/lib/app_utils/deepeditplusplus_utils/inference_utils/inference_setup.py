import logging
from typing import Callable, Sequence, Union

# from lib.transforms.transforms import GetCentroidsd
# from monailabel.deepeditPlusPlus.transforms import (
#     AddGuidanceFromPointsDeepEditd,
#     AddGuidanceSignalDeepEditd,
#     DiscardAddGuidanced,
#     ResizeGuidanceMultipleLabelDeepEditd,
#     AddSegmentationInputChannels,
#     IntensityCorrection,
#     MappingLabelsInDatasetd, 
#     MappingGuidancePointsd
# )
from monai.inferers import Inferer, SimpleInferer
# from monai.transforms import (
#     Activationsd,
#     AsDiscreted,
#     EnsureChannelFirstd,
#     EnsureTyped,
#     LoadImaged,
#     Orientationd,
#     Resized,
#     ScaleIntensityRanged,
#     SqueezeDimd,
#     ToNumpyd,
#     DivisiblePadd,
#     CenterSpatialCropd,
# )

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
# from monailabel.transform.post import Restored
##################################################

#Imports for the parametrised utils

from os.path import dirname as up
import sys
import os

deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from inference_utils.pre_transf_setup import run_get_inference_pre_transf
from inference_utils.post_transf_setup import run_get_inference_post_transf

# ################################################


logger = logging.getLogger(__name__)


class DeepEditPlusPlus(BasicInferTask):
    """
    This provides Inference Engine
    """

    def __init__(
        self,
        path,
        modality,
        infer_version_params, #The dict which contains the information about which version of each component is being used. 
        transforms_parametrisation_dict, #The dict which contains any information necessary for the transforms.
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=3,
        number_intensity_ch=1,
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )
        
        #######
        self.transforms_parametrisation_dict = transforms_parametrisation_dict #Parameters which may be used for the transforms (e.g. padding size)
        
    
        #Fixed parameters 
        self.number_intensity_ch = number_intensity_ch
        self.modality = modality 
        self.load_strict = False

        #Extracting the dict of component version params.
        self.version_params = infer_version_params 

        #Importing the version params for the components..
        self.pre_transf_version_param = self.version_params['pre_transforms_version_param']
        self.inferer_version_param = self.version_params['inferer_version_param'] 
        self.inverse_transf_version_param = self.version_params['inverse_transforms_version_param'] 
        self.post_transf_version_param = self.version_params['post_transforms_version_param'] 


        self.supported_pre_transform_versions = ['-3', '-2', '-1', '1','2']
        self.supported_inferer_versions = ['0']
        self.supported_inverse_transform_versions = ['0']
        self.supported_post_transform_versions = ['0']

        assert self.pre_transf_version_param in self.supported_pre_transform_versions 
        assert self.inferer_version_param in self.supported_inferer_versions 
        assert self.inverse_transf_version_param in self.supported_inverse_transform_versions
        assert self.post_transf_version_param in self.supported_post_transform_versions 
        
    def pre_transforms(self, data=None):
        return run_get_inference_pre_transf(dict(vars(self)), data, self.pre_transf_version_param)

    def inferer(self, data=None) -> Inferer:
        if self.inferer_version_param == '0':
            return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        
        if self.inverse_transf_version_param == '0':
            return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:

        return run_get_inference_post_transf(dict(vars(self)), data, self.post_transf_version_param)
