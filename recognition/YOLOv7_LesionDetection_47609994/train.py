# source code for training, val, test, saving model
# model imp from modules.py
# data loader imp from dataset.py

from modules import YOLOModel
import os

# Set paths
DATA_YAML = 'ISIC-2017.yaml'
WEIGHTS = 'yolov7.pt'
OUTPUT_DIR = 'outputs'
EPOCHS, IMGSZ, BATCH_SIZE = 50, 640, 8

os.makedirs(OUTPUT_DIR, exist_ok=True)  # make sure got output folder, handle crashing

# initialise model
model = YOLOModel(weights_path=WEIGHTS)

# train model
model.train(
    data_yaml=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE
)
