
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any, Optional

import torch
from torch.nn.modules.loss import _Loss

__all__ = ["MaskedLoss"]


class MaskedLoss(_Loss):
    """
    This is a wrapper class for the loss functions.  It allows for binary weighting masks to be applied to both input and target to mask out regions.

    This is only supported for loss functions which do not have any notion of distance implicitly required. (e.g. Hausdorff distance)

    This wrapper works by extracting the voxels (spatially) which correspond to the binary mask.
    See Also:
        
    """

    def __init__(
        self, 
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | _Loss,
        batchwise_reduction: str,
        *loss_args: Any, 
        **loss_kwargs: Any
    ) -> None:
        """
        Args:
            loss: loss function to be wrapped, this could be a loss class or an instance of a loss class.
            batchwise_reduction: a string which denotes what the reduction strategy is across the batch
            loss_args: arguments to the loss function's constructor if `loss` is a class.
            loss_kwargs: keyword arguments to the loss function's constructor if `loss` is a class.
        """
        super().__init__()
        self.loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            loss(*loss_args, **loss_kwargs) if inspect.isclass(loss) else loss
        )
        if not callable(self.loss):
            raise ValueError("The loss function is not callable.")

        self.batchwise_reduction = batchwise_reduction 

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should be B1H[WD].
        """
        if mask is None:
            # warnings.warn("No mask value specified for the MaskedLoss.")
            raise ValueError('No mask was specified for the MaskedLoss')
            # return self.loss(input, target)

        if input.dim() != mask.dim():
            warnings.warn(f"Dim of input ({input.shape}) is different from mask ({mask.shape}).")
        if input.shape[0] != mask.shape[0] and mask.shape[0] != 1:
            raise ValueError(f"Batch size of mask ({mask.shape}) must be one or equal to input ({input.shape}).")
        if target.dim() > 1:
            if mask.shape[1] != 1:
                raise ValueError(f"Mask ({mask.shape}) must have only one channel.")
            if input.shape[2:] != mask.shape[2:]:
                warnings.warn(f"Spatial size of input ({input.shape}) is different from mask ({mask.shape}).")

        #Extracting number of samples in batch:
        batch_size = input.shape[0] 

        if input.shape[0] != target.shape[0]:
            raise ValueError('The batch size should be consistent between pred and target')
        
        #Extracting number of classes
        n_ch = input.shape[1]
        
        #Extracting the shape of the spatial size of the input/target/mask. 
        spatial_size = input.shape[2:]
        num_dims = int(len(spatial_size))

        
        #We save the set of batch-wise generated losses and then implement reduction afterwards. It is done in this manner due to the fact that there may be
        #variation in the "size" (i.e. number of retained voxels) in the binary mask, and therefore it cannot be stacked together since N_voxels would differ.

        batchwise_losses = [] 

        for sample_index in range(batch_size):
                
            #Extracting the number of voxels in the binary mask being taken into consideration, so that we can initialise the tensor. 
            
            num_voxels = int(mask[sample_index].sum())

            #Extracting the new input, target masks with size 1(flattened) tensor, e.g. where the "flattened" tensor is a 1 x 1 x N or 1 x N dimensional tensor, 
            #where N = number of voxels/pixels that are in the retained binary mask.

            # new_dims = [1] + [n_ch] + [1] * (num_dims - 1) + [num_voxels]
            new_spatial_dims = tuple([1] * (num_dims - 1) + [num_voxels])
            # new_input = torch.zeros(new_dims)
            # new_target = torch.zeros(new_dims)

            new_inputs_list = []
            new_targets_list = []

            for ch in range(n_ch):
                #Indexing is correct!
                new_inputs_list.append(torch.reshape(torch.masked_select(input[sample_index, ch], mask[sample_index,0].detach().bool()), new_spatial_dims))
                new_targets_list.append(torch.reshape(torch.masked_select(target[sample_index, ch], mask[sample_index,0].detach().bool()), new_spatial_dims))


            
            #Reformatting the channel/class separated list into a 1N_ch11N_vox tensor.
            new_input = torch.unsqueeze(torch.stack(new_inputs_list), dim=0)
            new_target = torch.unsqueeze(torch.stack(new_targets_list), dim=0)

            if new_input.shape != new_target.shape:
                    raise ValueError('The shape of the input and target should be the same.')
                
            batchwise_losses.append(self.loss(new_input, new_target))
        
        if self.batchwise_reduction == 'Mean':
            return sum(batchwise_losses)/batch_size
        elif self.batchwise_reduction == 'Sum':
            return sum(batchwise_losses)
        # return self.loss(input * mask, target * mask) Deprecated version that only worked for dice. 