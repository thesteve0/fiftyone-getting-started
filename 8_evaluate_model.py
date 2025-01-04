import os
import tempfile
import torch
from ultralytics import YOLO
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

"""
Now are are going to run our fine tuned model on the validation data and see how we did.
I have already added the ground truth to the data so we can evaluate model performance

Good discussion of accuracy, precision, recall
https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
"""

classes = ["people", "no people"]

# This will point to the model we trained in the previous step
model = YOLO("./fine-tuning-yolo/train/weights/best.pt")
validation_data = load_from_hub("Voxel51/getting-started-validation-clip-pred", persistent=True, name="validation-clip-pred")

validation_data.apply_model(model, label_field="yoloft_predict")

# https://docs.voxel51.com/user_guide/evaluation.html#binary-evaluation
clip_results = validation_data.evaluate_classifications(
    "clip_predict",
    gt_field="ground_truth",
    eval_key="clip_eval",
    method="binary",
    classes=classes,
)

yoloft_results = validation_data.evaluate_classifications(
    "yoloft_predict",
    gt_field="ground_truth",
    eval_key="yoloft_binary",
    method="binary",
    classes=classes,
)


session = fo.launch_app(validation_data)
session.wait()

