import shutil
import os
from monailabel.utils.others.generic import file_ext

def label_saving(inference_res, output_dir, image_id, image_name_path):
    #Saving the labels: 
    label = inference_res["file"]
    label_json = inference_res["params"]
    #test_dir = os.path.join(output_dir , "labels", "final")
    os.makedirs(output_dir, exist_ok=True)

    label_file = os.path.join(output_dir, image_id + file_ext(image_name_path))
    shutil.move(label, label_file)


    print(label_json)
    print(f"++++ Image File: {image_name_path}")
    print(f"++++ Label File: {label_file}")
    
    return label_file