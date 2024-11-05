
'''The post transforms implemented in order to extract the segmentation mask 

Version 0: The original implementation in DeepEdit (original!).
'''
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureTyped,
    SqueezeDimd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from os.path import dirname as up
import sys
import os

deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from inference_transforms_utils.get_centroidsd import GetCentroidsd


def run_get_inference_post_transf(self_dict, data, func_version_param):

    if func_version_param == '0':
        return [
                EnsureTyped(keys="pred", device=data.get("device") if data else None),
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True), #just after this, all the dimensions get squeezed into one channel.
                SqueezeDimd(keys="pred", dim=0),
                ToNumpyd(keys="pred"), 
                Restored(
                    keys="pred",
                    ref_image="image",
                    config_labels=self_dict['labels'] if data.get("restore_label_idx", False) else None, 
                ),
                GetCentroidsd(keys="pred", centroids_key="centroids", version_param='0'),
            ]