# test driver script that calls and runs algorithm

from predict import load_latest_model, predict_image
import os

def test_driver():
    """
    evaluates the latest trained YOLOv8 model on test split
    """
    model = load_latest_model()
    if model is None:
        print("no trained model, try after training")
        return

    # evaluate on test set
    results = model.val(data="/Users/suyi/Desktop/3710/ISIC-2017/isic2017.yaml",
                        split="test",
                        imgsz=640,
                        batch=8,
                        verbose=True,
                        iou=0.8)  # min IoU >= 0.8 as per assignment question

    print(results)

    test_path = "/Users/suyi/Desktop/3710/ISIC-2017/test/images"
    if os.path.exists(test_path):
        example_image = os.path.join(test_path, os.listdir(test_path)[0])
        predict_image(model, example_image)


if __name__ == "__main__":
    test_driver()
