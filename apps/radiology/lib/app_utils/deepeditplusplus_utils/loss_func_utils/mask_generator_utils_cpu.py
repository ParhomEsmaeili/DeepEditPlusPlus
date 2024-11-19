import torch 
import numpy as np
from itertools import chain
import operator
import functools
from monai.utils import min_version, optional_import

# connected_comp_measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

class MaskGenerator:
    '''
    This mask generator assumes that for mask generation with click information, that at least one click must be placed among the classes. 
    And ideally for each of the classes (though not absolutely needed for this).
    
    Otherwise, the cross-class mask generation will completely fail.

    This is slightly different to the implementation in the score generation utils because temporal consistency mask does NOT require any information about changed 
    voxels in this scenario, and only will pertain to the click set.

    We also do not use per_class masks, only the cross class masks are relevant here.
    '''

    def __init__(self, click_map_types, human_measure): #, dict_of_class_codes): 
        
        assert type(click_map_types) == list, "Clicking weightmap types selected were not formatted as a list"
        # assert type(gt_map_types) == list, "GT-based weightmap types selected were not formatted as a list"
        assert type(human_measure) == str, "Human-centric measure selected was not formatted as a string"
        # assert type(dict_of_class_codes) == dict, "Dictionary of class integer codes was not formatted as such."
        
        # self.class_code_dict = dict_of_class_codes

        self.click_weightmap_types = [i.title() for i in click_map_types]
        #A list of the components of the weight-map which may originate solely from the clicks, e.g. a distance weighting, or an ellipsoid.

    
        self.human_measure = [human_measure] 
        #The measure of model performance being implemented, e.g. responsiveness in region of locality/non-worsening elsewhere. 

        self.supported_click_weightmaps = ['Ellipsoid',
                                            'Cuboid', 
                                            # 'Scaled Euclidean Distance',
                                            # 'Exponentialised Scaled Euclidean Distance',
                                            'Binarised Exponentialised Scaled Euclidean Distance',
                                            # '2D Intersections', 
                                            # 'None',
                                            ]
        
        
        self.supported_human_measures = ['Local Responsiveness',
                                        'Temporal Consistency',
                                        # 'None',
                                        ]

        if any([click_weightmap not in self.supported_click_weightmaps for click_weightmap in self.click_weightmap_types]):
            #Basic assumption is numbers and symbols will not be placed in the string, only potentially a string with non-capitalised words.
            raise Exception("Selected click-based weight map is not supported")
        
        
        if any([human_measure not in self.supported_human_measures for human_measure in self.human_measure]):
            raise Exception("Selected human-centric mask type is not supported")
        



    def click_based_weightmaps(self, guidance_points_set, guidance_point_parametrisations, include_background, image_dims):
        #The guidance points set is assumed to be a dictionary covering all classes, with each point being provided as a 2D/3D set of coordinates in a list.
        #Image dims are assumed to be a list.

        #Guidance point parametrisations is the nested dictionary which contains the parametrisations sorted by mask-type and then by class. 
        
        assert type(guidance_points_set) == dict, "The generation of click based weightmaps failed due to the guidance points not being in a dict"
        assert type(guidance_point_parametrisations) == dict, "The generation of click based weightmaps failed due to the guidance point parametrisations not being a dict"
        assert type(image_dims) == torch.Size, "The generation of click based weightmaps failed due to the image dimensions not being of a torch.Size class"
        assert type(include_background) == bool, "The generation of click based weightmaps failed due to the include_background parameter not being a bool: True/False"

        # for value in guidance_point_parametrisations.values():
        #     assert type(value) == dict, "The generation of click based weightmaps failed due to the parametrisations for each weightmap_type field not being a dict"
        
        

        # list_of_points = list(chain.from_iterable(list(guidance_points_set.values())))


        click_availability_bool = list(chain.from_iterable(list(guidance_points_set.values()))) != [] #Checking that the overall click set isn't empty
        
        if click_availability_bool == False:
            raise ValueError('There were no clicks across the classes even though a click based weightmap is being used.')

        # per_class_click_availability_bool = dict()    #Checking whether each click class is empty or not
        
        #We will obtain cross-class and per-class masks so that we generate cross-class fused, and per-class fused masks.

        cross_class_masks = []
        
        for item in self.click_weightmap_types:
            
            if item == "Ellipsoid":    
                
                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are pre-fused across intra-class points. Similarly for the cross-class one across all point masks.

                '''

                cross_class_mask = self.generate_ellipsoids(guidance_points_set,guidance_point_parametrisations[item], include_background, image_dims)


                cross_class_masks.append(cross_class_mask)

                # for key,val in per_class_mask.items():
                #     per_class_masks[key].append(val) 

            elif item == "Cuboid":
                

                '''
                We would like to generate a cross-class fused mask.

                Fused for the cross-class mask across all individual point masks.

                '''
                
                cross_class_mask = self.generate_cuboids(guidance_points_set, guidance_point_parametrisations[item], include_background, image_dims)

                cross_class_masks.append(cross_class_masks)

                # for key,val in per_class_mask.items():
                #     per_class_masks[key].append(val)
            
                
            elif item == "Binarised Exponentialised Scaled Euclidean Distance":
               

                #IF We only consider a single fused mask, need to flatten the guidance_point parameterisations for the given mask-level dict (i.e. the dict with class-point combinations).
                
                # list_of_guidance_point_parametrisations_dummy = list(chain.from_iterable(list(guidance_point_parametrisations[item].values())))
                
                '''
                We would like to generate a cross-class fused, and a per class fused set of masks.

                For each class, the masks are generated across intra-class points, exponentiated, then fused into one mask. Similarly for the cross-class one, but across all point masks.

                
                #We assume that the parametrisation also contains the information about the exponentiation parameter. 
                # And that it is the second to last parameter for each point's parametrisation.

                We also assume that a binarisation parameter is provided for binarising the probabilistic map.
                
                # Therefore it is length (n + 2) where n = the quantity of scaling parameters provided (1 or 2/3).
                
                #We assume that the exponentiation is only performed according to a single parametrisation. Any per dimension modification is to be done within the per-dimension scaling 
                #of the euclidean distance.

                '''

                # list_of_guidance_point_parametrisations = [sublist[:-1] for sublist in list_of_guidance_point_parametrisations_dummy]
                # list_of_exponentiation_parameters = [sublist[-1] for sublist in list_of_guidance_point_parametrisations_dummy]

                dict_of_scale_parametrisations = dict()
                dict_of_exponentiation_parametrisations = dict()
                binarisation_parameter = None 

                for class_label, list_of_point_parametrisations in guidance_point_parametrisations[item].items():

                    dict_of_scale_parametrisations[class_label] = [sublist[:-2] for sublist in list_of_point_parametrisations]
                    dict_of_exponentiation_parametrisations[class_label] = [sublist[-2] for sublist in list_of_point_parametrisations]

                    if binarisation_parameter == None:
                        
                        binarisation_parameter = list_of_point_parametrisations[0][-1]
                        for sublist in list_of_point_parametrisations[1:]:
                            assert sublist[-1] == binarisation_parameter
                        
                    else:

                        for sublist in list_of_point_parametrisations:
                            assert sublist[-1] == binarisation_parameter

                output_maps = self.generate_euclideans(True, dict_of_scale_parametrisations, guidance_points_set, include_background, image_dims, click_availability_bool, per_class_click_availability_bool, False)  

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing  lists of point separated masks.
                
                output_maps = self.exponentiate_map(dict_of_exponentiation_parametrisations, include_background, output_maps)

                assert type(output_maps) == dict
                #Output maps is a class-separated dict containing lists of point separated masks. 

                # print(guidance_points_set)
                fusion_strategy = "Additive"

                for key, map_list in output_maps.items():
                    
                    #We binarise this probabilistic map according to the binarisation parameter (i.e. the probabilistic value for binarisation)
                    binarised_fused_map = torch.where(self.map_fusion(fusion_strategy, map_list) > binarisation_parameter,1 , 0)
                    per_class_masks[key].append(binarised_fused_map) #We fuse the intra-class point masks.

                #We flatten all of the point masks into a list and fuse into one mask for the cross-clask mask.
                flattened = list(chain.from_iterable(list(output_maps.values() )))
                fused_map = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()])
                #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES!

                cross_class_masks.append(torch.where(fused_map > binarisation_parameter, 1, 0)) #We append the binarised map  
            
            

        
        #We then fuse across the cross-class list of masks, and the intra-class lists of masks.

        cross_class_fused = self.map_fusion('Multiplicative',cross_class_masks)
        # per_class_fused = dict()

        # for class_label, sublist in per_class_masks.items():
        #     per_class_fused[class_label] = self.map_fusion('Multiplicative', sublist)

        assert type(cross_class_fused) == torch.Tensor
        # assert type(per_class_fused) == dict 

        # return (cross_class_fused, per_class_fused) 
        return cross_class_fused
    
    
    def human_measure_weightmap(self, click_based_weightmaps, image_dims):
        '''        
        The click based weightmaps are a tuple: cross_class_fused weightmap which is fused across all of the selected click-based types
        
        The human_measure_information is any information required for the human_measure weightmap.
        '''

        assert type(click_based_weightmaps) == torch.Tensor 
        

        if self.human_measure[0].title() == "Local Responsiveness":
            #If locality is the measure then no change required. 
            cross_class_click_weightmap = click_based_weightmaps
            
            final_cross_class_weightmap = cross_class_click_weightmap #* cross_class_gt_weightmap 
            



        elif self.human_measure[0].title() == "Temporal Consistency":
            #If temporal consistency, then we need to invert a locality based mask.
            
            cross_class_click_weightmap = click_based_weightmaps
            
            final_cross_class_weightmap = 1 - cross_class_click_weightmap #* cross_class_gt_weightmap * cross_class_information

    
        # elif self.human_measure[0].title() == "None":
        #     #This is just for default scores! No weightmap
        #     final_cross_class_weightmap = torch.ones(image_dims) 
            
                

        assert type(final_cross_class_weightmap) == torch.Tensor 
        # assert type(per_class_final_weightmaps) == dict 
        
        return final_cross_class_weightmap#, per_class_final_weightmaps
    


    def __call__(self, guidance_points_set, point_parametrisations, include_background, image_dims):
        

        '''
        In instances where a weightmap subtype is not being used, the "None" will be the only corresponding selection in that list for the mask generator definition.

        Therefore, in these instances it will just generate tensors of ones.

        Cross class masks are generated across all classes (including background if specified)
        
        Args:

        guidance_points_set: The set of guidance points across the classes positions.
        point_parametrisations: The corresponding set of parametrisations for the selected mask type for each click.
        include_background: Whether the background is to be included in the generation of masks.
        image dims: The dimensions of the image in the orientation that the clicks were placed in. 
       

        Returns:

        The cross-class mask for the given image and its clicks (or lackthereof) according to the mask type and the parametrisation.
        
        '''
        
        
        assert type(guidance_points_set) == dict
        
        '''Ensure that the value for each class key (class name) in the dict is a nested list of the points which are represented as a list of coordinates'''

        for guidance_point_list in guidance_points_set.values():

            if any([type(entry) != list for entry in guidance_point_list]):
                raise Exception("Non-list entry in the list of guidance points")

        '''Also takes a dictionary which contains the parameterisations for each mask type selected AND for each point.
        
        The structure is therefore a nested dictionary:
            - Mask type
                - Parametrisation in the same structure as the guidance points dict.
        
                    I.e., Each class in the guidance points has a nested list containing the parameterisation for that MASK TYPE AND CLICK.
        
        '''

        #Asserting dictionary at the upper level
        assert type(point_parametrisations) == dict, "Input point parametrisation was not a dictionary"

        #Asserting that each value must also be a dictionary.
        for value_entry in point_parametrisations.values():
            assert type(value_entry) == dict, "Mask-level structure in point parameterisation was not a dictionary"

            #Asserting that each value in the dictionary must be a nested list for each segmentation class.  
            for class_point_parametrisations in value_entry.values():
                assert type(class_point_parametrisations) == list, "Class-level structure in point parametrisation was not a list"

                for point_level_parametrisation in class_point_parametrisations:
                    assert type(point_level_parametrisation) == list, "Point-level structure in point parametrisation was not a list"
        

        ''' Also takes a list containing the dimensions of the image, for which the weight-maps will be created, assumed to be in RAS orientation for 3D images'''

        assert type(image_dims) == torch.Size, "Image dimensions were not in a torch.Size datatype"
        
        assert type(include_background) == bool, "Information about including the background was not provided in the bool format."

        # assert type(gt) == torch.Tensor or gt == None, "The ground truth provided was not in the right format, torch.Tensor or NoneType"
        
        
        
        cross_class_click_weightmaps = self.click_based_weightmaps(guidance_points_set, point_parametrisations, include_background, image_dims)

        # click_weightmaps = (cross_class_click_weightmaps, per_class_click_weightmaps)
            
            
        # cross_class_gt_weightmaps, per_class_gt_weightmaps = self.gt_based_weightmaps(guidance_points_set, include_background, image_dims, gt)

        # gt_weightmaps = (cross_class_gt_weightmaps, per_class_gt_weightmaps)

        cross_class_map = self.human_measure_weightmap(cross_class_click_weightmaps, image_dims)
        
        assert type(cross_class_map) == torch.Tensor 
        # assert type(per_class_maps) == dict 
         
        # return cross_class_map, per_class_maps 

        return cross_class_map


    def map_fusion(self, fusion_strategy, maps):
        '''
        Map fusion function which fuses together a LIST of maps either by pure additive fusion, elementwise multiplication, or by finding the union of booleans
        '''
        supported_fusions = ["Additive", "Multiplicative", "Union"]
        
        assert fusion_strategy.title() in supported_fusions, "Selected fusion strategy is not supported by the image map fusion function"

        if  fusion_strategy == "Additive":
            summed_output_maps = sum(maps)
            
            return summed_output_maps/torch.max(summed_output_maps) 
        
        if fusion_strategy == "Multiplicative":

            product_output_maps = functools.reduce(operator.mul, maps, 1)
            return product_output_maps/torch.max(product_output_maps)
        
        if  fusion_strategy == "Union":

            union_output_maps = sum(maps)
            return torch.where(union_output_maps > 0, 1, 0)

    def exponentiate_map(self, dict_of_exponentiation_parameters, include_background, maps):
        '''
        Returns class-separated dict of lists of point-masks.
        '''
        output_maps = dict()

        for class_label, parametrisation in dict_of_exponentiation_parameters.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 
        
            class_maps = maps[class_label]

            output_maps[class_label] = [torch.exp(-parametrisation[i] * weight_map) for i,weight_map in enumerate(class_maps)]
        
        return output_maps

    def generate_euclideans(self, is_normalised_bool, scaling_parametrisations_set, guidance_points_set, include_background, image_dims, square_root_bool):
        
        '''Is_normalised parameter just assesses whether the distances are scaled by the scaling parametrisations
           Axis scaling parametrisation is the scaling denominator of the summative terms of the euclidean computation.
           squre_root_bool just assesses whether square root the maps or not
        '''
        assert type(is_normalised_bool) == bool, "Is_normalised bool parameter in euclidean map generation was not a bool"
        # assert type(additive_fusion_bool) == bool, "Additive Fusion bool parameter in euclidean map generation was not a bool"
        assert type(guidance_points_set) == dict, "Generation of euclidean map failed because points were not in a class-separated dict"
        assert type(image_dims) == torch.Size, "Generation of euclidean map failed because the image dimension provided was not torch.Size datatype"
        assert type(scaling_parametrisations_set) == dict, "Generation of euclidean map failed because the axis scaling parametrisation was not a within a class-separated dict"
        # assert type(click_avail_bool) == bool 
        # assert type(per_class_click_avail_bool) == dict 

        per_class_masks = dict() 
        # full_set_of_masks = []
        
        for class_label, list_of_points in guidance_points_set.items():

            if not include_background:
                if class_label.title() == "Background":
                    continue 
            
            list_of_scaling_parametrisation = scaling_parametrisations_set[class_label]

            centres = [[coord + 0.5 for coord in centre] for centre in list_of_points]

            
            intra_class_masks = []

            if len(centres) == 0:
                #No clicks for the given class

                intra_class_masks = [torch.ones(image_dims) * torch.nan]
            
            else: #There were clicks for the given class.
                for i, centre in enumerate(centres):
                    
                    assert type(list_of_scaling_parametrisation[i]) == list, "Generation of euclidean map failed because the axis scaling parametrisation for each point was not a list"

                    intra_class_masks.append(self.each_euclidean(is_normalised_bool, list_of_scaling_parametrisation[i], centre, image_dims, square_root_bool))

            

            per_class_masks[class_label] = intra_class_masks

            # full_set_of_masks += intra_class_masks
            


        # assert type(cross_class_mask) == torch.Tensor 
        assert type(per_class_masks) == dict 

        return per_class_masks
    
    def each_euclidean(self, is_normalised, scaling_parametrisation, point, image_dims, square_root_bool):
        
        '''Is_normalised parameter just assesses whether the distances are scaled by a scaling parametrisation'''
        assert type(is_normalised) == bool, "Is_normalised parameter in euclidean map generation was not a bool"
        assert type(point) == list, "Generation of euclidean map failed because point was not a list"
        assert type(image_dims) == torch.Size, "Generation of euclidean map failed because the image dimension provided was not torch.Size datatype"
        assert type(scaling_parametrisation) == list, "Scaling parametrisation for the denom terms of the euclidean were not provided in the list format"
        assert type(square_root_bool) == bool, 'Square root bool was not a bool'

        if len(scaling_parametrisation) == 1:
            scaling_parametrisation*= len(image_dims)
        else:
            pass

        grids = [torch.linspace(0.5, image_dim-0.5, image_dim) for image_dim in image_dims]
        meshgrid = torch.meshgrid(grids, indexing='ij')

        if square_root_bool:
            if is_normalised:
                return torch.sqrt(sum([torch.square((meshgrid[i] - point[i])/(scaling_parametrisation[i])) for i, image_dim in enumerate(image_dims)]))
            else:
                return torch.sqrt(sum([torch.square(meshgrid[i] - point[i]) for i in range(len(image_dims))]))
        else:

            if is_normalised:
                return sum([torch.square((meshgrid[i] - point[i])/(scaling_parametrisation[i])) for i, image_dim in enumerate(image_dims)])
            else:
                return sum([torch.square(meshgrid[i] - point[i]) for i in range(len(image_dims))])

    def generate_cuboids(self, guidance_points_set, scale_parametrisation_set, include_background, image_dims):
        '''
        Cuboids require parameterisation.

        Parametrisation is a set of raw parameters for each point.

        This parametrisation is the raw quantity of voxels..(e.g. 50 voxels in x, 75 in y, 90 in z) because we might have variations in the actual physical measurement per voxel (e.g. 1 x 10 x 10mm)
        
        Returns:

        Cross-class fused mask and a dict of per-class fused masks across the guidance points correspondingly.
        '''
        
        assert type(scale_parametrisation_set) == dict, "Structure of scale parametrisations across classes in cuboid generator was not a dict (with nested lists)"
        assert type(guidance_points_set) == dict, "Structure of guidance point sets across classes in cuboid generator was not a dict (with nested lists)"
        assert type(include_background) == bool 
        assert type(image_dims) == torch.Size, "Datatype for the image dimensions in cuboid generator was not torch.Size"
        # assert type(click_avail_bool) == bool 
        # assert type(per_class_click_avail_bool) == dict 

        for list_of_scale_parametrisation in scale_parametrisation_set.values():
            assert type(list_of_scale_parametrisation) == list, "Structure of scale parametrisations for a given class in cuboid generator was not a list"

            for sublist in list_of_scale_parametrisation:

                assert type(sublist) == list, "Structure of scale parametrisations for each point in cuboid generator was not a list"

        


        #For each class, we generate the per-class fused mask. 
        per_class_masks = dict()

        for class_label, list_of_points in guidance_points_set.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 
            
            #Initialising the mask:
            mask = torch.zeros(image_dims)


            #Extracting the list of scale parametrisations for the set of points of the given class. 

            list_of_scale_parametrisation = scale_parametrisation_set[class_label]


            for point, scale_parametrisation in zip(list_of_points, list_of_scale_parametrisation):

                #shifting the centre to the center of each voxel (where we assume click placement occurs)
                centre = [coord + 0.5 for coord in point]


                '''
                None of the scale parameterisations should be larger than the 0.5 of the corresponding image dimensions otherwise the box would be larger than the image.
                '''


                if len(scale_parametrisation) == 1:
                    parametrisation = scale_parametrisation * len(image_dims)
                else:
                    parametrisation = scale_parametrisation

                if any(torch.tensor(parametrisation)/torch.tensor(image_dims) > 0.5):
                    raise Exception("Scale factors for the cuboid size mean that the dimensions would be larger than the image")
                
                
                #obtain the extreme points of the cuboid which will be assigned as the box region:
                min_maxes = []


                for index, coordinate in enumerate(centre):
                    #For each coordinate, we obtain the extrema points.
                    dimension_min = int(max(0, torch.round(torch.tensor(coordinate - parametrisation[index]))))
                    dimension_max = int(min(image_dims[index] - 1, torch.round(torch.tensor(coordinate + parametrisation[index]))))

                    min_max = [dimension_min, dimension_max] 
                    min_maxes.append(min_max)


                if len(image_dims) == 2:
                #If image is 2D            
                    mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1]] = 1
                elif len(image_dims) == 3:
                    #If image is 3D:
                    mask[min_maxes[0][0]:min_maxes[0][1], min_maxes[1][0]:min_maxes[1][1], min_maxes[2][0]:min_maxes[2][1]] = 1

            # if per_class_click_avail_bool[class_label] == False: #In the scenario where the click set is completely empty for this class. 

            #     mask = torch.ones(image_dims) * torch.nan 

            #If the click set is empty then we just pass an empty tensor, because we are only computing cross class masks we do not need the nan generator.

            #Appending the "fused" mask into the per-class mask dict.
            per_class_masks[class_label] = mask 

        #Fusing the per-class masks into a single cross class mask also.
        fusion_strategy = "Union"

        #We placed the per-class masks into a list and fuse into one mask for the cross-clask mask.
        flattened = list(per_class_masks.values() )
        cross_class_mask = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()]) 
        #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! Should not be any anyways, for empty click sets
        #for a given class we just left it with an empty zero tensor.


        assert type(cross_class_mask) == torch.Tensor 
        # assert type(per_class_masks) == dict 

        return cross_class_mask


    def generate_ellipsoids(self, guidance_points_set, scale_parametrisations_set, include_background, image_dims):
        
        '''
        #Ellipsoid requires parametrisation: There are three options available
        

        #For each point, the following parametrisation configurations are permitted: 

        #param_1 only: All dimensions have the same scaling

        #param_1/2 or param_1/2/3 indicate separate scalings

        #In instances where it has separate scalings, this is assumed to be a list with length > 1! 

        #All parameters must be smaller than the resolution of the corresponding dimensions.

        #Mask is a torch tensor.


        Inputs: Guidance points sets, split by class. Scale parametrisations set for the guidance points. Whether the background is included: Bool. Image dimensions in the same orientation of the 
        guidance points.

        Returns:

        Fused mask of ellipsoids across the classes corresponding to the guidance points that were provided.
        '''
    
        assert type(image_dims) == torch.Size, "Image dimensions for the ellipsoid mask generators were not of the torch.Size type"
        assert type(scale_parametrisations_set) == dict, "scale parametrisation for the ellipsoid mask generators were not of the dict datatype"
        assert type(include_background) == bool
        assert type(image_dims) == torch.Size 
        # assert type(click_avail_bool) == bool 
        # assert type(per_class_click_avail_bool) == dict 

        per_class_masks = dict() 
        fusion_strategy = 'Union'

        for class_label, list_of_points in guidance_points_set.items():
            
            if not include_background:
                if class_label.title() == "Background":
                    continue 

            #Generate the per-class fused mask

            list_of_scale_parametrisation = scale_parametrisations_set[class_label] 
            
            assert type(list_of_scale_parametrisation) == list, "List of scale parametrisations in the ellipsoid generator was not a nested list for each class"

            #Collect the set of ellipsoid binary masks
            ellipsoid_masks = []

            for point, scale_parametrisation in zip(list_of_points, list_of_scale_parametrisation):
                
                if len(scale_parametrisation) == 1:
                    parametrisation = scale_parametrisation * len(image_dims)
                else:
                    parametrisation = scale_parametrisation

                if any(torch.tensor(parametrisation)/torch.tensor(image_dims) > 0.5):
                    raise Exception("Scale factor too large, axis of ellipse will be larger than the image")

                assert type(scale_parametrisation) == list, "Scale parametrisation for each point for ellipsoid generation was not in a list structure"

                
                ellipsoid_masks.append(self.each_ellipsoid(point, parametrisation, image_dims))
                
            if list_of_points == []: #If there is no click set!
                ellipsoid_masks = [torch.ones(image_dims) * torch.nan] #We put a dummy nan in here.

            per_class_masks[class_label] = self.map_fusion(fusion_strategy, ellipsoid_masks)

        #We placed the per-class masks into a list and fuse into one mask for the cross-clask mask.
        flattened = list(per_class_masks.values() )

        cross_class_mask = self.map_fusion(fusion_strategy, [i for i in flattened if not i.isnan().any()]) 
        #we remove any of the nan tensors for cross class mask computation, WE ASSUME AT LEAST ONE CLICK ACROSS CLASSES! There should not be any anyways, for the empty
        #sets we just leave the tensors empty.

        # cross_class_mask = self.map_fusion(fusion_strategy, list(chain.from_iterable(list(per_class_masks.values() ))))

        assert type(cross_class_mask) == torch.Tensor 
        # assert type(per_class_masks) == dict 
         
        return cross_class_mask


    def each_ellipsoid(self,centre, scale_factor_denoms, image_dims):
        #Generating the bool mask outlining the ellipsoid defined using the centre, and the scale parameters corresponding to each image dimension. This ranges from [0, 0.5]
        
        
        #Ellipsoids is defined in the following manner: (x-xo/a)^2 + (y-yo/b)^2 + (z-zo/c)^2 = 1 (for 2D the z-term is just dropped)

        #We treat point coordinates as being at the center of each voxel.  
        
        #We create a grid of coordinates for the image:
        grids = [torch.linspace(0.5, image_dim - 0.5, image_dim) for image_dim in image_dims]

        #shifting the centre 
        centre = [coord + 0.5 for coord in centre]

        #computing the denominators using the scale_factors for each image_dimension
        denoms =  scale_factor_denoms #[scale_factor_denoms[i] for i,image_dim in enumerate(image_dims)]
        
        #generating the coordinate set
        
        # if len(image_dims) == 2:
        #     meshgrid = torch.meshgrid(grids[0], grids[1], indexing='ij')
        # else:
        #     meshgrid = torch.meshgrid(grids[0], grids[1], grids[2], indexing='ij')   

        meshgrid = torch.meshgrid(grids, indexing='ij')
        
        lhs_comp = sum([torch.square((meshgrid[i] - centre[i])/denoms[i]) for i in range(len(image_dims))])

        return torch.where(lhs_comp <= 1, 1, 0)