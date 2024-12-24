import fiftyone as fo
import fiftyone.zoo as foz


"""
For this workshop we are going to only load a zoo dataset. If you want to load your own data,
start by looking at the import section of the documentation
https://docs.voxel51.com/user_guide/dataset_creation/index.html
"""

# https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset

dataset = foz.load_zoo_dataset("caltech101", persistent=True, overwrite=True)

print("just plain print: " + str(dataset) + "\n\n")
print("Same as a summary: " + dataset.summary() + "\n\n")

clone_dataset = dataset.clone("cloned")
print("Current datasets: " + str(fo.list_datasets()))
