# source code for training, validating, testing, saving
# model imported from modules.py
# data pre-processed in dataset.py by running it once, no further actions
# losses and metrics plotted during training

from modules import load_model

def train_and_val(num_epochs):
    """
    loads YOLOv8 model, trains, and validates
    """

    # load YOLOv8 with pre-trained weights
    model = load_model("/Users/suyi/Desktop/3710/yolov8n.pt")  # using: yolov8n.pt

    # training
    model.train(
        data="/Users/suyi/Desktop/3710/ISIC-2017/isic2017.yaml",  # path to yaml
        epochs=num_epochs,
        imgsz=640,
        batch=8,
        name="lesion_detector",  # name of run
        project="runs/train",
        verbose=True
    )

    metrics = model.val()
    print(metrics)  # get accuracy score etc


if __name__ == "__main__":
    train_and_val(20)
