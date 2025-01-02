import os
import tempfile
import torch
from ultralytics import YOLO
import fiftyone as fo

"""
This code will show you how to fine tune a YOLO model with the data we worked with. We are not going to run it in the workshops
since that would take too long. This is here for when you want to see this later
"""

DATASET_NAME = 'training_data'
DEFAULT_MODEL_SIZE = "m"
DEFAULT_IMAGE_SIZE = 320
DEFAULT_EPOCHS = 5
PROJECT_NAME = 'fine-tuning-yolo'



def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_classifier(
        dataset_name=None,
        model_size=DEFAULT_MODEL_SIZE,
        image_size=DEFAULT_IMAGE_SIZE,
        epochs=DEFAULT_EPOCHS,
        project_name="mislabel_confidence_noise",
        gt_field="ground_truth",
        train_split=None,
        test_split=None,
        **kwargs
):

    # settings.update({"wandb": False})
    if dataset_name:
        dataset = fo.load_dataset(dataset_name)
        dataset.take(0.2 * len(dataset)).tag_samples("test")
        dataset.match_tags("test", bool=False).tag_samples("train")
        train = dataset.match_tags("train")
        test = dataset.match_tags("test")
    else:
        train = train_split
        test = test_split

    if model_size is None:
        model_size = "s"
    elif model_size not in ["n", "s", "m", "l", "x"]:
        raise ValueError("model_size must be one of ['n', 's', 'm', 'l', 'x']")

    splits_dict = {
        "train": train,
        "val": test,
        "test": test,
    }

    data_dir = tempfile.mkdtemp()

    for key, split in splits_dict.items():
        split_dir = os.path.join(data_dir, key)
        os.makedirs(split_dir)
        split.export(
            export_dir=split_dir,
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=gt_field,
            export_media="symlink",
        )

    # Load a pre-trained YOLOv8 model for classification
    model = YOLO(f"yolo11{model_size}-cls.pt")

    # Train the model
    model.train(
        data=data_dir,  # Path to the dataset
        epochs=epochs,  # Number of epochs
        imgsz=image_size,  # Image size
        device=get_torch_device(),
        project=project_name,
    )

    return model


def main():    

    train_classifier(
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
    )


if __name__ == "__main__":
    main()