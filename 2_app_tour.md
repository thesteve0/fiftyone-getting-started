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

# Codespaces doesn't play well with websockets or SSE so at the end of our URLs we need to put
# ?polling=true
# Which adds long polling
```

Things to do
1. Look at the datasets available
2. Left side and what you can do with the fields
3. selecting images - show how that appears in the API and vice versa
https://docs.voxel51.com/api/fiftyone.core.session.html?highlight=session#fiftyone.core.session.Session
```
selected_samples = session.selected

session.select_samples(["677e09e04b4ee07986bd0c79"])

```
4. looking at an individual image
3. Tag labels or samples in the app
4. Lightning = indexed field. To turn off toggle query performance mode
5. Filter gear
6. Change color schemes
7. Plugins 

https://docs.voxel51.com/plugins/index.html

https://github.com/voxel51/fiftyone-plugins 

**Plugin plugin**
```
fiftyone plugins download https://github.com/voxel51/fiftyone-plugins --plugin-names @voxel51/plugins
```
**Image Quality**
data quality - There is a teams feature that does this and more https://docs.voxel51.com/teams/data_quality.html

```
fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```
refresh the page then [ctrl] + `, then "common issues". Let's do brightness

**Build a dashboard** https://github.com/voxel51/fiftyone-plugins/tree/main/plugins/dashboard

Groundtruth histogram and then scatterplot of Uniqueness versus Brightness

```
fiftyone plugins download https://github.com/voxel51/fiftyone-plugins     --plugin-names @voxel51/dashboard
```

8. Finally, you can change the data in the view from the API. 

```python
session.dataset = dataset.limit(5).clone('little_data')
```