from ultralytics import YOLO

# load model
def load_model(path):
    """
    loads YOLOv8 model from ultralytics package, handles loading error

    Args:
        path (str): path to file from ultralytics Git
    """
    try:
        model = YOLO(path)
        return model
    except ModuleNotFoundError:
        print("Failed to load model")
        return None

