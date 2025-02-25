import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

"""
We now have the basic objects and app interface under our belt. Time to start doing real world work. 
Let's start examining and prepping our data for analysis. 
We are going to import an unlabeled set of images, calculate embeddings, and then explore what the embeddings 
teach us about our data.
"""

PHOTO_DIR = "/photos"
DATASET_NAME = "photos"

# downloaded 443 public domain images from flickr, just images with no metadata. Load the data using the "directory of images" importer
# https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#imagedirectory-import

dataset = fo.Dataset.from_dir(
    dataset_dir=PHOTO_DIR,
    dataset_type=fo.types.ImageDirectory,
    name=DATASET_NAME,
    overwrite=True,
    persistent=True
)


# By default it uses mobilenet-v2-imagenet-torch" 
# https://docs.voxel51.com/model_zoo/models.html#mobilenet-v2-imagenet-torch
# Trained on imagenet
fob.compute_visualization(dataset, embeddings="default_embed", brain_key="default_embed")

# Trained on 400 Million image text pairs from the internet
model = foz.load_zoo_model("open-clip-torch")
fob.compute_visualization(dataset, model=model, embeddings="open_clip_embed",
                              brain_key="openclip_embed")

# Based on a DETR model trainied on COCOA Data
model2 = foz.load_zoo_model("detection-transformer-torch")
fob.compute_visualization(dataset, model=model2, embeddings="det_transformer", brain_key="det_ransformer")

# trained on imagenet
# model2 = foz.load_zoo_model("mnasnet0.5-imagenet-torch")
# fob.compute_visualization(dataset, model=model2, embeddings="mnasnet_embed",
                            #   brain_key="mnasnet_embed")

# Calculate representativness
# https://docs.voxel51.com/brain.html#image-representativeness
fob.compute_representativeness(dataset, progress=True)

# Calculate uniqueness
# https://docs.voxel51.com/brain.html#image-uniqueness
fob.compute_uniqueness(dataset, embeddings="open_clip_embed")

# Calculate near duplicates
# https://docs.voxel51.com/brain.html#near-duplicates
index = fob.compute_near_duplicates(
    dataset,
    embeddings="open_clip_embed",
    # may need to change this distance measure for non-default mode: thresh=0.02,
    )

duplicates_view = index.duplicates_view(
    type_field="dup_type",
    id_field="dup_id",
    dist_field="dup_dist",
)


session = fo.launch_app(dataset)
session.wait()

print("done")


