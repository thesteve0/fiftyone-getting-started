from requests import sessionfrom requests import session

# Walk through for the application

There will be limited code during this section so we will just use the terminal

```shell
python3
```

```python
import fiftyone as fo

fo.list_datasets()

dataset = fo.load_dataset("quickstart")

session = fo.launch_app(dataset)

#If we ran this in code we would need to hold the session open to prevent the app server from exiting
# session.wait()
```

Things to do
1. Look at the datasets available
2. Left side and what you can do with the fields
3. selecting images - show how that appears in the API and vice versa
4. looking at an individual image
3. Tag labels or samples
4. Lightning = indexed field. To turn off toggle query performance mode
5. Filter gear
6. Change color schemes
7. Plugins - image quality
8. Plugins - build a dashboard


```python
session.dataset = fo.load_dataset("cloned")
```