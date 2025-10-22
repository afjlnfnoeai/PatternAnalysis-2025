# data loader for loading and preprocessing data
import os
import cv2

dataset = "/Users/suyi/Desktop/3710/ISIC-2017"  # path to dataset on local hardware

# test, train, val respective folders
types = ["test", "train", "val"]

for i in types:
    ea_type_data = os.path.join(dataset, i)

    images_dir = os.path.join(ea_type_data, "images")  # create sub-folders
    labels_dir = os.path.join(ea_type_data, "labels")
    os.makedirs(images_dir, exist_ok=True)  # make sure don't crash
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(ea_type_data):
        if not file.endswith("_superpixels.png"):
            continue

        # file paths
        mask_path = os.path.join(ea_type_data, file)
        img_name = file.replace("_superpixels.png", ".jpg")
        img_path = os.path.join(ea_type_data, img_name)

        # image + mask, mask in grayscale & og in colour
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)

        if img is None or mask is None:  # skip any missing files
            print(f"Skipped {img_name}; image/mask not found in {i}")
            continue  # make sure don't crashing

        # find image size, lesion's contours
        h, w = img.shape[:2]
        contours, ignore = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotations = []

        # bounding boxes
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)

            # normalised coordinates according to YOLO standards
            x_center, y_center = ((bx + bw /2)/w), ((by + bh /2)/h)
            width, height = (bw/w), (bh/h)

            annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # YOLO images file
        dst_img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(dst_img_path):
            os.rename(img_path, dst_img_path)

        # YOLO label file
        label_file = os.path.join(labels_dir, file.replace("_superpixels.png", ".txt"))
        with open(label_file, "w") as f:  # all bounding boxes into new file
            f.write("\n".join(annotations))

    print(f"labels generated for {i}")

