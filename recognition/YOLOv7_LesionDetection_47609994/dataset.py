# data loader for loading and preprocessing data
import os
import cv2

dataset = "/Users/suyi/Desktop/3710/ISIC-2017"

# test, train, val respective folders
types = ["ISIC-2017_Test_v2_Data",
         "ISIC-2017_Training_Data",
         "ISIC-2017_Validation_Data"]

for i in types:
    ea_type_data = os.path.join(dataset, i)

    images_dir = os.path.join(ea_type_data, "images")  # create sub-folders
    labels_dir = os.path.join(ea_type_data, "labels")
    os.makedirs(images_dir, exist_ok=True)  # handle crashing
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(ea_type_data):
        if not file.endswith("_superpixels.png"):
            continue

        # file paths
        mask_path = os.path.join(ea_type_data, file)
        img_name = file.replace("_superpixels.png", ".jpg")
        img_path = os.path.join(ea_type_data, img_name)

