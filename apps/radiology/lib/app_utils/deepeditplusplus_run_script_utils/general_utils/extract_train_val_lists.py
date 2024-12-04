import json
import os
import sys
from os.path import dirname as up


def load_data_split_lists(train_folds, val_fold, dataset_name):

    #Extracting the directory path for the overall codebase
    codebase_directory = up(up(up(up(up(up(up(os.path.abspath(__file__))))))))

    ########## Loading in the list of train/val images #######################

    dataset_dir_outer = os.path.join(codebase_directory, "datasets", dataset_name) 
    #The dataset folder which contains alllll of the information, and not just the imagesTr folder.

    with open(os.path.join(dataset_dir_outer, "train_val_split_dataset.json")) as f:
        dictionary_setting = json.load(f)
        val_dataset = dictionary_setting[f"fold_{val_fold}"]
        training_dataset = []
        for i in train_folds:
            # if i != int(val_fold):
            training_dataset += dictionary_setting[f"fold_{i}"]

    ########## Joining the subdir/image-labels     
    for pair_dict in val_dataset:
        pair_dict["image"] = os.path.join(dataset_dir_outer, pair_dict["image"][2:]) + '.nii.gz'
        pair_dict["label"] = os.path.join(dataset_dir_outer, pair_dict["label"][2:]) + '.nii.gz'

    for pair_dict in training_dataset:
        pair_dict["image"] = os.path.join(dataset_dir_outer, pair_dict["image"][2:]) + '.nii.gz'
        pair_dict["label"] = os.path.join(dataset_dir_outer, pair_dict["label"][2:]) + '.nii.gz'

    return training_dataset, val_dataset