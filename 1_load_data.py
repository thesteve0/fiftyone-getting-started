import fiftyone as fo
import fiftyone.zoo as foz


"""
For this workshop we are going to only load a zoo dataset. If you want to load your own data,
start by looking at the import section of the documentation
https://docs.voxel51.com/user_guide/dataset_creation/index.html
"""

# https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset

print("Current datasets: " + str(fo.list_datasets())+ "\n\n")

# Let's start by loading a zoo dataset
# https://docs.voxel51.com/dataset_zoo/datasets.html
dataset = foz.load_zoo_dataset("quickstart", persistent=True)

print("just plain print: " + str(dataset) + "\n\n")
print("Same as a summary: " + dataset.summary() + "\n\n")

print("Current datasets: " + str(fo.list_datasets()) + "\n\n")

print("Number of samples: " + str(dataset.count()) + "\n\n")

print("Dataset schema: " + str(dataset.get_field_schema()) + "\n\n")

sample = dataset.first()
print(str(sample.field_names) + "\n\n")

# To show non-persistence without a save run this file again but comment out the next line
# Or open a python terminal load the same dataset and then see what the value is for .first()
sample["steve_field"] = True

print(str(sample.field_names) + "\n\n")

sample.save()

print("Tried to add steve_field - Dataset schema: " + str(dataset.get_field_schema()) + "\n\n")

dataset.add_sample_field("general_field", fo.ListField, subfield=fo.StringField)

# Fields are strongly typed. To see the error, uncomment this line and run again
# sample2 = dataset.last()
# sample2["steve_field"] = 2

print("Tried to add general_field - Dataset schema: " + str(dataset.get_field_schema()) + "\n\n")

# Now views versus cloned datasets
dataset_view = dataset.view()
if "cloned" in fo.list_datasets():
    fo.delete_dataset("cloned")

clone_dataset = dataset.clone("cloned")

##### TODO make some changes to schema or value in the view and see how things change. Same does not happen in a clone
# you can not alter the dataset schema through the view. Ucomment to see the error
# dataset_view.add_sample_field("on_the_view", fo.IntField)

# But you can change sample data values through the view just like a dataset
# It has SOME of the same methods and properties. It can also be used in place of a dataset in some use cases.
# We will cover them a bit later. JUst be careful when you do operations that can change values
# Datasets are lightweight - get used to using clones

dsview_sample = dataset_view.first()
dsview_sample["general_field"] = ["dataset field updated through view"]
dsview_sample.save()


cloned_sample = clone_dataset.first()
cloned_sample["general_field"] = ["cloned sample field"]
cloned_sample.save()

print("Dataset first: " + str(dataset.first()["general_field"]) + "\n\n")
print("Now Cloned Dataset first: " + str(clone_dataset.first()["general_field"]) + "\n\n")


print("Done with API intro")
