# source code for training, validating, testing, saving
# model imported from modules.py
# data pre-processed in dataset.py by running it once, no further actions
# losses and metrics plotted during training

from modules import load_model
import os
import glob

def train_and_val(num_epochs):
    """
    loads YOLOv8 model, trains, and validates
    """

    # load YOLOv8 with pre-trained weights
    # assignment sheet stated that pre-trained mod els are allowed
    model = load_model("/Users/suyi/Desktop/3710/yolov8n.pt")  # using: yolov8n.pt, will auto download if don't have

    # training
    model.train(
        data="/Users/suyi/Desktop/3710/ISIC-2017/isic2017.yaml",  # path to yaml
        epochs=num_epochs,
        imgsz=640,
        batch=8,
        name="lesion_detector",  # name of run
        project="runs/train",  # will create folder if it doesn't yet exist
        verbose=True
    )

    metrics = model.val()
    print(metrics)  # get accuracy score etc

def test(training_path):
    """
    finds most recent trained model, evaluate on test split

    Args:
        training_path (str): path to saved trained files in project
    """

    # get most recent training fun
    training_folders = sorted(glob.glob(os.path.join(training_path, "*")),
                         key=os.path.getmtime,
                         reverse=True)

    if not training_folders:
        print("no training runs yet")
        return

    # get model just trained
    latest = training_folders[0]
    best_model = os.path.join(latest, "weights", "best.pt")

    if not os.path.exists(best_model):
        print("not found")
        return

    print(f"latest trained model loaded from {best_model}")

    # eval on test split
    model = load_model(best_model)
    results = model.val(
        data="/Users/suyi/Desktop/3710/ISIC-2017/isic2017.yaml",
        split="test",
        imgsz=640,
        batch=8,
        verbose=True,
        project="runs/vals",
        iou = 0.8  # question wants all detections iou >= 0.8
    )

    print(results)


if __name__ == "__main__":
    train_and_val(50)  # adjust accordingly to num of epochs u want
    test("runs/train")
