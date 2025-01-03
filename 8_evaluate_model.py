import os
import tempfile
import torch
from ultralytics import YOLO
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

"""
Now are are going to run our fine tuned model on the validation data and see how we did.
I have already added the ground truth to the data so we can evaluate model performance


"""

# This will point to the model we trained in the previous step
model = YOLO("./fine-tuning-yolo/train/weights/best.pt")




validation_data = load_from_hub("Voxel51/getting-started-labeled-validation", persistent=True, name="labeled_validation_photos")

