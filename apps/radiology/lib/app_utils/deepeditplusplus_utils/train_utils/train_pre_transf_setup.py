'''
File which contains the function which produces the compose list of transforms for the pre transform component in the train setup. 
'''

import logging

import torch

from os.path import dirname as up
import os
import sys
deepeditpp_utils_dir = up(up(os.path.abspath(__file__)))
sys.path.append(deepeditpp_utils_dir)

from transforms_utils.add_guidance_signald import AddGuidanceSignalDeepEditd
from transforms_utils.add_initial_seed_point_missing_labelsd import AddInitialSeedPointMissingLabelsd
from transforms_utils.add_random_guidanced import AddRandomGuidanceDeepEditd
from transforms_utils.find_all_valid_slices_missinglabelsd import FindAllValidSlicesMissingLabelsd
from transforms_utils.find_discrepancy_regionsd import FindDiscrepancyRegionsDeepEditd
from transforms_utils.splitpredlabelsd import SplitPredsLabeld
from transforms_utils.normalize_labels_in_datasetd import NormalizeLabelsInDatasetd
from transforms_utils.add_segmentation_input_channelsd import AddSegmentationInputChannelsd
from transforms_utils.extract_meta_informationd import ExtractMetad
# from transforms_utils.mapping_labels_from_configd import MappingLabelsInDatasetd

from transforms_utils.modality_based_normalisationd import ImageNormalisationd
from transforms_utils.randblurd import UniformRandGaussianSmoothd
from transforms_utils.rand_gamma_adjustd import RandGammaAdjustnnUNetd
from transforms_utils.rand_gaussian_noise_nnu_netd import RandGaussianNoisennUNetd
from transforms_utils.brightness_adjustd import RandBrightnessd 
from transforms_utils.contrast_adjustd import RandContrastAdjustd


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
    CenterSpatialCropd,
    RandSimulateLowResolutiond
)

logger = logging.getLogger(__name__)

def run_get_train_pre_transf(self_dict, context, func_version_param):

    assert type(self_dict) == dict 
    assert type(func_version_param) == str 

    supported_version_params = ['-7', '-6', '-5', '-4', '-3', '-2', '-1', '1','2', '3']
    
    assert func_version_param in supported_version_params, 'The version parameter was not supported for the get train pre-transform list composition'

    if func_version_param == '-7':
        #Modification to the augmentation stack, in order to be more nnu-net like. Contains support for additional augmentations to the input image provided in the nnu-net stack
        #including: gaussian noise injection.

        #Parametrisation for the currently provided transforms are set to be the same as the nnu-net implementation.

        t = [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'), 
            RandGaussianNoisennUNetd(keys="image", sample_var= True, version_param = '1'),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50), 
            #Here we will pad the image to fit the requirements of the backbone architecture, .
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            #Beware: This rotation is only appropriate in the instance where the two initial spatial dimension axes are the same shape. Otherwise the actual image
            #region is being translated.
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")),
        ]
        
        return t

    elif func_version_param == '-6':
        #Modification to compare zero padding and padding which mirrors the background better (e.g. edge padding)
        t = [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'), 
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50), 
            #Here we will pad the image to fit the requirements of the backbone architecture, .
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor'], mode='edge'),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            #Beware: This rotation is only appropriate in the instance where the two initial spatial dimension axes are the same shape. Otherwise the actual image
            #region is being translated.
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")),
        ]

        return t 

    if func_version_param == '-5':
        '''Deprecated'''
        #Modification to the augmentation stack, in order to be more nnu-net like. Contains support for additional augmentations to the input image provided in the nnu-net stack
        #including: gaussian noise injection, gaussian blurring, contrast and brightness adjustment, gamma correction. Hyperparams are set to be nnu-net like.

        #For this version we inject all the modifications except for the image degradation, still has a bug.. We also add back the rand shift intensity, it appears the
        #contrast adjustments aren't configured well without a patch based method.... (too much background?)
        # t = [
        #     LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
        #     EnsureChannelFirstd(keys=("image", "label")),
        #     NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
        #     Orientationd(keys=["image", "label"], axcodes="RAS"),
        #     ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'),
        #     # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
        #     # RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
        #     RandGaussianNoisennUNetd(keys="image", prob= 0.1, mean=0.0, var_bound = (0, 0.1), sample_var= True, version_param = '1'),
        #     UniformRandGaussianSmoothd(keys="image", kernel_bounds=(0.5, 1.5), prob = 0.1, version_param = '1'),
        #     RandBrightnessd(keys="image", prob= 0.15, bounds = (0.7, 1.3), version_param='1'),
        #     RandContrastAdjustd(keys="image", prob= 0.15, bounds= (0.65, 1.5), preserve_range = True, version_param = '1'),
        #     RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        #     RandGammaAdjustnnUNetd(keys="image", gamma_no_inv = (0.7, 1.5), gamma_with_inv = (0.7, 1.5), no_inv_gamma_prob = 0.1, with_inv_gamma_prob = 0.3, retain_stats = True, version_param = '1'),
        #     #Here we will pad the image to fit the requirements of the backbone architecture. Zero padding is used after intensity aug. so that the model learns to 
        #     #ignore this region easier, rather than learning to ignore pecularities about stochastic augmentations in the padding region.
        #     DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']), 
        #     RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        #     RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        #     RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        #     #Beware: This rotation is only appropriate in the instance where the two initial spatial dimension axes are the same shape. Otherwise the actual image
        #     #region is being translated.
        #     RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        #     ToTensord(keys=("image", "label")),
        #     SelectItemsd(keys=("image", "label", "label_names")),
        # ]

        return t 

    if func_version_param == '-4':
        '''Deprecated'''
        #Modification to the augmentation stack, in order to be more nnu-net like. Contains support for additional augmentations to the input image provided in the nnu-net stack
        #including: gaussian noise injection, gaussian blurring, contrast and brightness adjustment, gamma correction. Hyperparams are set to be nnu-net like.

        #For this version we inject all the modifications except for the image degradation, still has a bug..
        # t = [
        #     LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
        #     EnsureChannelFirstd(keys=("image", "label")),
        #     NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
        #     Orientationd(keys=["image", "label"], axcodes="RAS"),
        #     ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'),
        #     # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
        #     # RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
        #     RandGaussianNoisennUNetd(keys="image", prob= 0.1, mean=0.0, var_bound = (0, 0.1), sample_var= True, version_param = '1'),
        #     UniformRandGaussianSmoothd(keys="image", kernel_bounds=(0.5, 1.5), prob = 0.1, version_param = '1'),
        #     RandBrightnessd(keys="image", prob= 0.15, bounds = (0.7, 1.3), version_param='1'),
        #     RandContrastAdjustd(keys="image", prob= 0.15, bounds= (0.65, 1.5), preserve_range = True, version_param = '1'),
        #     RandGammaAdjustnnUNetd(keys="image", gamma_no_inv = (0.7, 1.5), gamma_with_inv = (0.7, 1.5), no_inv_gamma_prob = 0.1, with_inv_gamma_prob = 0.3, retain_stats = True, version_param = '1'),
        #     #Here we will pad the image to fit the requirements of the backbone architecture. Zero padding is used after intensity aug. so that the model learns to 
        #     #ignore this region easier, rather than learning to ignore pecularities about stochastic augmentations in the padding region.
        #     DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']), 
        #     RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        #     RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        #     RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        #     #Beware: This rotation is only appropriate in the instance where the two initial spatial dimension axes are the same shape. Otherwise the actual image
        #     #region is being translated.
        #     RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        #     ToTensord(keys=("image", "label")),
        #     SelectItemsd(keys=("image", "label", "label_names")),
        # ]

        return t 

    if func_version_param == '-3':
        #Modification to the augmentation stack, shifting the intensity augmentations to occuring prior to padding so that the "ignored region" doesn't have stochastic augs. 
        
        t = [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            #Intensity modif is applied prior to padding so that the model does not have to deal with stochasticity in the region it should ignore.

            #Here we will pad the image to fit the requirements of the backbone architecture. Zero padding is used so that intensity augmentations can avoid concerning
            #themselves with issues caused by padding (e.g. shift of the mean for contrast adjustment).
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']), 
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")),
        ]

        return t 
    if func_version_param == '-2':
        #Modification to the normalisation strategy to use z-score for non x-ray, and use the heuristic planner vals for clipping the x-ray based modalities.
        #Also modifies the padding so that it uses the edge value for padding (i.e. background value). 
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", planner_dict = context.planner_dict, modality = self_dict['modality'], version_param='4'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor'], mode='edge'),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")),
        ]

    elif func_version_param == '-1':
        #Basic version (-1) for a simple U-net model, in order to check that the pre-processing/i.e. the non editing/interactivity components are reasonable.
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='2'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")),
        ]


    if func_version_param == '1':
            
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='1'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]
    
    elif func_version_param == '2':

        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='2'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]
    
    elif func_version_param == '3':
        #Modified version of version_2 but with the added modification that the click set is propagated to the inner loop again. 
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            # MappingLabelsInDatasetd(keys="label", original_label_names=self_dict['original_dataset_labels'], label_names = self_dict['_labels'], label_mapping=self_dict['label_mapping']),
            NormalizeLabelsInDatasetd(keys="label", label_names=self_dict['_labels'], version_param='0'), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ImageNormalisationd(keys="image", modality = self_dict['modality'], version_param='2'),
            #Here we will pad the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self_dict['component_parametrisation_dict']['divisible_padding_factor']),
            # CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            # Transforms for click simulation for interactive init.
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids",version_param='0'),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids",version_param='0'),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self_dict['number_intensity_ch'], version_param='1'),
            AddSegmentationInputChannelsd(keys="image", previous_seg_name= None, number_intensity_ch = self_dict['number_intensity_ch'], label_names=None, previous_seg_flag= False, version_param='1'),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "guidance", "label_names")),
        ]
