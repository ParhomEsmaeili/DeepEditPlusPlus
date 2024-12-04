from __future__ import annotations

import warnings
from itertools import chain

import numpy as np
import torch
import time
from monai.config import KeysCollection
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import concatenate
from monai.utils import PostFix, convert_data_type, convert_to_tensor

DEFAULT_POST_FIX = PostFix.meta()


class nnUNetLikeDatasetSummary:
    """
    This class provides a modification to the implementation of the Dataset Summary class by Monai, in order to be more nnU-net like.
    
    This allows for the extraction of information related to voxel spacing according to
    the input dataset. The extracted values can used to resample the input in segmentation tasks
    (like using as the `pixdim` parameter in `monai.transforms.Spacingd`).

    In addition, it also supports to compute the mean, std, min and max intensities of the input,
    and these statistics are helpful for image normalization

    The algorithm for calculation refers to:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.

    """

    def __init__(
        self,
        dataset: Dataset,
        image_key: str | None = "image",
        label_key: str | None = "label",
        meta_key: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        num_workers: int = 0,
        **kwargs,
    ):
        """
        Args:
            dataset: dataset from which to load the data.
            image_key: key name of images (default: ``image``).
            label_key: key name of labels (default: ``label``).
            meta_key: explicitly indicate the key of the corresponding metadata dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, affine, original_shape, etc.
                if None, will try to construct meta_keys by `{image_key}_{meta_key_postfix}`.
                This is not required if `data[image_key]` is a MetaTensor.
            meta_key_postfix: use `{image_key}_{meta_key_postfix}` to fetch the metadata from dict,
                the metadata is a dictionary object (default: ``meta_dict``).
            num_workers: how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process (default: ``0``).
            kwargs: other parameters (except `batch_size` and `num_workers`) for DataLoader,
                this class forces to use ``batch_size=1``.

        """

        self.data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=num_workers, **kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.meta_key = meta_key or f"{image_key}_{meta_key_postfix}"
        self.all_meta_data: list = []

        #This part below is taken from nnU-net preprocessing: 

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)

        self.num_foreground_voxels_for_intensity_stats = 10e7

        if type(dataset.data) != list:
            raise TypeError("The datatype for the dataset object's data list needs to be a list.")
        
        if len(dataset.data) == 0:
                raise ValueError('There are currently no images available in the data list for sampling!')


    def collect_meta_data(self):
        """
        This function is used to collect the metadata for all images of the dataset. Input data is assumed to be a MetaTensor initially loaded using LoadImaged
        """

        for data in self.progressBar(self.data_loader, prefix='Progress', suffix='Complete', length=50):

            if not isinstance(data, dict):
                raise TypeError('Data loader requires dictionary based transforms to be passed through the dataloader')
            
            meta_dict = {}

            if isinstance(data[self.image_key], MetaTensor):
                meta_dict = data[self.meta_key]
                
                #We assert that the data must be single modality for our current application!
                if int(meta_dict['pixdim[4]'][0]) > 1:
                    raise ValueError('This application only supports single modality implementations.')

            else:
                raise KeyError(f"To collect metadata for the dataset, `{self.meta_key}` must exist.")
            
            self.all_meta_data.append(meta_dict)

            time.sleep(0.1)

    def get_target_spacing(self, spacing_key: str = "affine", anisotropic_threshold: int = 3, percentile: float = 10.0):
        # """
        # Calculate the target spacing according to all spacings.
        # If the target spacing is very anisotropic,
        # decrease the spacing value of the maximum axis according to percentile.
        # The spacing is computed from `affine_to_spacing(data[spacing_key][0], 3)` if `data[spacing_key]` is a matrix,
        # otherwise, the `data[spacing_key]` must be a vector of pixdim values.

        # Args:
        #     spacing_key: key of the affine used to compute spacing in metadata (default: ``affine``).
        #     anisotropic_threshold: threshold to decide if the target spacing is anisotropic (default: ``3``).
        #     percentile: for anisotropic target spacing, use the percentile of all spacings of the anisotropic axis to
        #         replace that axis.

        # """
        # if len(self.all_meta_data) == 0:
        #     self.collect_meta_data()
        # if spacing_key not in self.all_meta_data[0]:
        #     raise ValueError("The provided spacing_key is not in self.all_meta_data.")
        # spacings = []
        # for data in self.all_meta_data:
        #     spacing_vals = convert_to_tensor(data[spacing_key][0], track_meta=False, wrap_sequence=True)
        #     if spacing_vals.ndim == 1:  # vector
        #         spacings.append(spacing_vals[:3][None])
        #     elif spacing_vals.ndim == 2:  # matrix
        #         spacings.append(affine_to_spacing(spacing_vals, 3)[None])
        #     else:
        #         raise ValueError("data[spacing_key] must be a vector or a matrix.")
        # all_spacings = concatenate(to_cat=spacings, axis=0)
        # all_spacings, *_ = convert_data_type(data=all_spacings, output_type=np.ndarray, wrap_sequence=True)

        # target_spacing = np.median(all_spacings, axis=0)
        # if max(target_spacing) / min(target_spacing) >= anisotropic_threshold:
        #     largest_axis = np.argmax(target_spacing)
        #     target_spacing[largest_axis] = np.percentile(all_spacings[:, largest_axis], percentile)

        # output = list(target_spacing)

        # return tuple(output)
        raise NotImplementedError('Still need to implemement the spacings extractor for image resampling')

    def calculate_intensity_statistics(self, foreground_threshold: int = 0, random_seed: int = 1234, num_samples: int = 10000):
        """
        This function is used to calculate the maximum, minimum, mean and standard deviation of intensities of
        the input dataset.

        Args:
            foreground_threshold: the threshold to distinguish if a voxel belongs to foreground, this parameter
                is used to select the foreground of images for calculation. Normally, `label > 0` means the corresponding
                voxel belongs to foreground, thus if you need to calculate the statistics for whole images, you can set
                the threshold to ``-1`` (default: ``0``).

        """
        voxel_sum = torch.as_tensor(0.0)
        voxel_square_sum = torch.as_tensor(0.0)
        voxel_max, voxel_min = [], []
        voxel_ct = 0

        for data in self.data_loader:
            if self.image_key and self.label_key:
                image, label = data[self.image_key], data[self.label_key]
            else:
                image, label = data
            image, *_ = convert_data_type(data=image, output_type=torch.Tensor)
            label, *_ = convert_data_type(data=label, output_type=torch.Tensor)

            image_foreground = image[torch.where(label > foreground_threshold)]

            voxel_max.append(image_foreground.max().item())
            voxel_min.append(image_foreground.min().item())
            voxel_ct += len(image_foreground)
            voxel_sum += image_foreground.sum()
            voxel_square_sum += torch.square(image_foreground).sum()

        self.data_max, self.data_min = max(voxel_max), min(voxel_min)
        self.data_mean = (voxel_sum / voxel_ct).item()
        self.data_std = (torch.sqrt(voxel_square_sum / voxel_ct - self.data_mean**2)).item()

    def calculate_intensity_percentiles(
        self,
        foreground_threshold: int = 0,
        sampling_flag: bool = True,
        interval: int = 10,
        min_percentile: float = 0.5,
        max_percentile: float = 99.5,
    ):
        """
        This function is used to calculate the percentiles of intensities (and median) of the input dataset. To get
        the required values, all voxels need to be accumulated. To reduce the memory used, this function can be set
        to accumulate only a part of the voxels.

        Args:
            foreground_threshold: the threshold to distinguish if a voxel belongs to foreground, this parameter
                is used to select the foreground of images for calculation. Normally, `label > 0` means the corresponding
                voxel belongs to foreground, thus if you need to calculate the statistics for whole images, you can set
                the threshold to ``-1`` (default: ``0``).
            sampling_flag: whether to sample only a part of the voxels (default: ``True``).
            interval: the sampling interval for accumulating voxels (default: ``10``).
            min_percentile: minimal percentile (default: ``0.5``).
            max_percentile: maximal percentile (default: ``99.5``).

        """
        all_intensities = []
        for data in self.data_loader:
            if self.image_key and self.label_key:
                image, label = data[self.image_key], data[self.label_key]
            else:
                image, label = data
            image, *_ = convert_data_type(data=image, output_type=torch.Tensor)
            label, *_ = convert_data_type(data=label, output_type=torch.Tensor)

            intensities = image[torch.where(label > foreground_threshold)].tolist()
            if sampling_flag:
                intensities = intensities[::interval]
            all_intensities.append(intensities)

        all_intensities = list(chain(*all_intensities))
        self.data_min_percentile, self.data_max_percentile = np.percentile(
            all_intensities, [min_percentile, max_percentile]
        )
        self.data_median = np.median(all_intensities)

    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        Images=image with single channels = shape (1, x, y(, z))
        """
        assert images.ndim == 4 and segmentation.ndim == 4
        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0
        percentiles = np.array((0.5, 50.0, 99.5))

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])

            mean, median, mini, maxi, percentile_99_5, percentile_00_5 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            if num_fg > 0:
                percentile_00_5, median, percentile_99_5 = np.percentile(foreground_pixels, percentiles)
                mean = np.mean(foreground_pixels)
                mini = np.min(foreground_pixels)
                maxi = np.max(foreground_pixels)

            intensity_statistics_per_channel.append({
                'mean': mean,
                'median': median,
                'min': mini,
                'max': maxi,
                'percentile_99_5': percentile_99_5,
                'percentile_00_5': percentile_00_5,

            })

        return intensities_per_channel, intensity_statistics_per_channe

    @staticmethod
    def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iterable    - Required  : iterable object (Iterable)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)
        # Progress Bar Printing Function
        def printProgressBar (iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Initial Call
        printProgressBar(0)
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        # Print New Line on Complete
        print()