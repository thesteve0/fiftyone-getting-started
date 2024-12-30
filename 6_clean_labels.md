# Cleaning labels and ground truth

We are going to do a little data cleaning and prep but then load the data I finished cleaning. Given the length of the workshop, cleaning the labels would take too long to do together.
All the samples will eventually have a ground truth label of "people" or "no people"

We will start by using tags to mark missed predictions with their eventual label

This process is more interactive and so we won't be running a Python file. Instead we will be using a combination of 
the REPL and the FiftyOne Application. 

Start python in the terminal below:
```
python
```

Now handle our imports, load the data, and launch the app:

```
import fiftyone as fo

dataset = fo.load_dataset("predicted_labels", persistence=True)
session = fo.launch_app(dataset)
```

Then, in the app find a mislabeled sample and check the box in the top left corner of the picture
