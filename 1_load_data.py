import fiftyone as fo
import fiftyone.zoo as foz


"""
For this workshop we are going to only load a zoo dataset. If you want to load your own data,
start by looking at the import section of the documentation
https://docs.voxel51.com/user_guide/dataset_creation/index.html
"""

# https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset

dataset = foz.load_zoo_dataset("quickstart", persistent=True)

print("Current datasets: " + str(fo.list_datasets()))
print("just plain print: " + str(dataset) + "\n\n")
print("Same as a summary: " + dataset.summary() + "\n\n")

if "cloned" in fo.list_datasets():
    fo.delete_dataset("cloned")

clone_dataset = dataset.clone("cloned")
print("Current datasets: " + str(fo.list_datasets()))

print("Number of samples: " + str(dataset.count()))

print("Dataset schema: " + str(dataset.get_field_schema()))

sample = dataset.first()
sample.field_names

print("Done with API intro")
