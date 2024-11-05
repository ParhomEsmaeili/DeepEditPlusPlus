from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Hashable, Mapping, Sequence, Sized

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import
from monai.transforms import ScaleIntensityRange, ScaleIntensityRangePercentiles, ScaleIntensity, NormalizeIntensity, ClipIntensityPercentiles

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)

distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


'''

Transform which propagates the prior segmentations as additional input channels (after the image, and N classes of point guidance tensors):

Version 1: The DeepEdit++ v1.1 implementation. 
Version 2: The DeepEdit++ v1.1 implementation in TORCH. 
'''

class AddSegmentationInputChannelsd(Randomizable, MapTransform):
    '''
    Generates the additional channels to concatenate with the image tensor after the guidance channels, representing the "previous segmentation" split by class.

    Inputs:
    Previous seg name is just a string which contains the name of the prior segmentation to call from the data dictionary.
    Number intensity channel is just the quantity of channels in the input image (we will only be working with single modality/modality-sequence data)
    
    Label names: The class config dictionary which contains the class label names and the class integer codes. This is the one which has been processed already
    within the pre-processing transforms (although this should not differ from the config labels in the txt file once the label mapping is removed).

    Previous seg flag: Flag which asserts whether a previous segmentation exists to propagate.
    version_param: The parameter which controls which version of this transform we are using. 
    '''
    def __init__(self, keys: KeysCollection, 
                allow_missing_keys: bool = False, 
                previous_seg_name:str | None = None, 
                number_intensity_ch : int = 1, 
                label_names: dict | None = None, 
                previous_seg_flag: bool = False,
                version_param: str = '1'):
        super().__init__(keys, allow_missing_keys)
        
        self.previous_seg_name = previous_seg_name
        self.number_intensity_ch = number_intensity_ch
        self.label_names = label_names or {}
        self.previous_seg_flag = previous_seg_flag
        #self.previous_seg = previous_seg 
        self.version_param = version_param 

        self.supported_versions = ['1']

        assert self.version_param in self.supported_versions

    def randomize(self, image):
        """
        Random generation of the initialisation segmentations. 
        """
        
        if self.version_param == '1':
            random_array = self.R.choice(list(self.label_names.values()), image.shape[1:])
            # print(np.unique(random_array, return_counts=True))
            return random_array
            
        #raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


    def _get_mask(self, image, previous_seg):
        '''
        get the mask according to the previous_Segmentation_flag, if in the Editing mode then it is the previous segmentation, if it is the other two modes 
        (AutoSeg/Interactive Init) then it should be randomly generated..

        previous segmentation is assumed to be a single tensor or None, not k-channels separated by class, for Editing this function splits the input image. 
        we assume that the tensor has discrete values which represent the integer codes for the classes (discrete) 
        
        (Current assumption is that the config label-integer codes are pre-normalised so that they go from values 0 - k-1).
        
        Similarly, the previous segmentation image is imported with values of 0 - k-1.

        '''
        
        if self.version_param == '1':

            output_signal = [] #initialise the list which we will concatenate to. 
            
            if self.previous_seg_flag:
                
                for key in self.label_names.keys():
                    output_signal.append(np.where(previous_seg == self.label_names[key], 1, 0)) 
                    
                return np.stack(output_signal, dtype=np.float32)    
                    
            else:
                #TODO Generate a HWD array where each voxel is randomly sampling uniformly from classes 1:k with probabilty 1/k. Use logical arrays to split into k channel tensors.
                random_array = self.randomize(image)
                
                for key in self.label_names.keys():
                    output_signal.append(np.where(random_array == self.label_names[key], 1, 0))
                    
                return np.stack(output_signal, dtype=np.float32)
        

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)

        if self.version_param == '1':

            if "will_interact" in d.keys():
                #If there a "will_interact" entry in dict then use the one from the data dictionary. This is intended to be used for the inner loop of training.
                will_interact = d["will_interact"]
            else: 
                #If there is not, then use the default that it will interact. This is used for the pre-transforms and inference, since we always want guidance to be added
                will_interact = True

            if will_interact:
                for key in self.key_iterator(d):
                    if key == "image":
                        image = np.copy(d[key])
                        
                        n_dims = len(image[0].shape)
                        
                        if self.previous_seg_flag: #If in inference for example , where the previous segmentation should still be a (n + 1)D tensor where n = spatial dimensions of image:  
                            if len(d[self.previous_seg_name].shape) == n_dims + 1:
                                previous_seg = d[self.previous_seg_name].squeeze()
                            elif len(d[self.previous_seg_name].shape) == n_dims: #If the previous segmentation is already an n * D tensor where n = spatial dimensions of image
                                previous_seg = d[self.previous_seg_name]
                            #TODO: Delete this temporary check.    
                            #nib.save(nib.Nifti1Image(np.array(previous_seg), None), os.path.join('/home/parhomesmaeili/TrainingInnerLoopPrediction/ActivatedPred.nii.gz'))
                        else:
                            previous_seg = None


                        #if label names is not inputted when instantiating the class, use the label names from the data dictionary.
                        if self.label_names:
                            pass
                        else:
                            self.label_names = d["label_names"]    
                        
                        tmp_image = image[0 : self.number_intensity_ch + len(self.label_names), ...]
                
                        # Getting signal
                        signal = self._get_mask(image, previous_seg) 

                        #logger.info(f"Dimensions of the split channels are {signal.shape}")
                        
                        tmp_image = np.concatenate([tmp_image, signal], axis=0, dtype=np.float32)
                        
                        # for i in range(tmp_image.shape[0]):
                        #     placeholder = tmp_image[i]
                        #     nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/AddingSegmentationChannels', str(i)+'.nii.gz'))

                        if isinstance(d[key], MetaTensor):
                            d[key].array = tmp_image
                        else:
                            d[key] = tmp_image
                        return d
                    else:
                        print("This transform only applies to image key")
            return d