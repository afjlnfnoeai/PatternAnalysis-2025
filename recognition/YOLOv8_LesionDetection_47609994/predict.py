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

