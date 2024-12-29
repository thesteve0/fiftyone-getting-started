import fiftyone as fo
import fiftyone.core.dataset as Dataset
import fiftyone.brain as fob
from fiftyone import ViewField as F

"""
We will look at multiple ways to sample and subset your data. This can be at the data import stage all the way through using expressions
on calculated fields
"""
PHOTO_DIR = "/photos"

# We could have done it at data import with random sampling
random_import_subset = fo.Dataset.from_dir(
    dataset_dir=PHOTO_DIR,
    dataset_type=fo.types.ImageDirectory,
    name="random_import_subset",
    overwrite=True,
    max_samples = 300,   
    shuffle = True,
    seed = 0.5

)

# We could also do it is a call on an existing dataset
full_photos = fo.load_dataset("photos")

# let's make a random clone so we don't mess with the original dataset
if Dataset.dataset_exists("random_clone_subset"):
    fo.delete_dataset("random_clone_subset")
random_clone_subset = full_photos.shuffle(seed=0.5).limit(300).clone("random_clone_subset", persistent = True)
fob.compute_visualization(random_clone_subset, embeddings="open_clip_embed", brain_key="random_clip_embed")

# Now let's actually subset using uniqueness and representativeness 
if Dataset.dataset_exists("unique_repr_clone_subset"):
    fo.delete_dataset("unique_repr_clone_subset")
unique_repr_clone_subset = full_photos.match((F("representativeness") > 0.5) & (F("uniqueness") > 0.5)).limit(300).clone("unique_repr_clone_subset", persistent = True)
fob.compute_visualization(unique_repr_clone_subset, embeddings="open_clip_embed", brain_key="unique_clip_embed")

# Now pick the rest of the photos for validation of our fine tune model
if Dataset.dataset_exists("validation_photos"):
    fo.delete_dataset("validation_photos")
final_validation = full_photos.exclude_by("filepath", unique_repr_clone_subset.values("filepath")).clone("validation_photos", persistent=True)

### TODO may want to add some command line playing with selections and having them update in the app
session = fo.launch_app()
session.wait()
