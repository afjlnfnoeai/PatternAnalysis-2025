# source code of the components of your model
# component must be implemented as a class/function

from ultralytics import YOLO

class YOLOModel:
    def __init__(self, weights_path='yolov7.pt'):
        """
        Initialize YOLOv7

        Args:
            weights_path: path to YOLOv7 weights
        """
        self.model = YOLO(weights_path)

    def train(self, data_yaml, epochs=50, imgsz=640, batch=8):
        """
        Train model

        Args:
            data_yaml: path to your YAML dataset config
            epochs: number of training epochs
            imgsz: image size
            batch: no. batches
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch
        )

    def predict(self, imgs, conf=0.25, save=False, save_dir='outputs/'):
        """
        Run inference on images

        Args:
            imgs: path to images/ list of image paths
            conf: confidence threshold
            save: save results Y/N
            save_dir: save output in folder
        """
        results = self.model.predict(imgs, conf=conf, save=save, save_dir=save_dir)
        return results
