# show example usage of trained model
# print out results/ provide visualisations

from modules import load_model
import os
import glob

def load_latest_model():
    """
    finds and gets most recent trained model
    """

    training_folders = sorted(glob.glob(os.path.join("runs/train", "*")),
                              key=os.path.getmtime,
                              reverse=True)

    if not training_folders:
        print("no training runs yet")
        return None

    latest_run = training_folders[0]
    best_model = os.path.join(latest_run, "weights", "best.pt")

    if not os.path.exists(best_model):
        print("not found")
        return None

    # get model
    return load_model(best_model)

def predict_image(model_to_use, source_path):
    """
    get predictions using latest model, display and save results

    Args:
        model_to_use: loaded model from load_latest_model(), or another model
        source_path: path to folder or image to predict
    """

    if model_to_use is None:
        return

    # run predictions
    results = model_to_use.predict(source=source_path,
                                   imgsz=640,
                                   conf=0.25,
                                   save=True,
                                   show=False,  # TRUE TO OPEN WINDOW WITH IMAGE
                                   project="runs/predict",
                                   name="lesion_predictions")

    for r in results:
        print("\nImage:", r.path)
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                
                print(f"class: {model_to_use.names[cls_id]}")
                print(f"confidence: {conf:.2f}")
                print(f"box: {xyxy}")
        else:
            print("no lesions detected")
