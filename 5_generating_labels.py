import fiftyone as fo
import fiftyone.core.dataset as Dataset
import fiftyone.zoo as foz

"""
We are going to take the non-random dataset from before and use the CLIP Model to generate some initial labels for the dataset
The problem we are trying to solve is "Are there people in the photograph or not"
"""

dataset = fo.load_dataset("unique_repr_clone_subset")

# The predictions will be saved on model run and we want to keep our original sample clean
if Dataset.dataset_exists("predicted_labels"):
    fo.delete_dataset("predicted_labels")
working_dataset = dataset.clone(name="predicted_labels", persistent=True)

clip = foz.load_zoo_model(
        "open-clip-torch",
        text_prompt="A photo ",
        classes= ["with a man", "with a woman", "with people", "without any people"])
        #classes= ["a person", "people", "no people"])

working_dataset.apply_model(clip, label_field="prediction")

session = fo.launch_app(working_dataset)

session.wait()
