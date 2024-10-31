# VIAM RE-ID OBJECT TRACKER

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for tracking object using ReID.


![example](img/output.gif)

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/modular-resources/#configuration) and select the `viam:vision:re-id-object-tracker` model from the [`re-id-object-tracker` module](https://app.viam.com/module/re-id-object-tracker).
This module implements the following methods of the [vision service API](https://docs.viam.com/services/vision/#api):

- `GetDetections()`: returns the bounding boxes with the unique id as label and the object detection confidence as confidence.
- `GetClassifications()`: returns the label `new_object_detected` for an image when a new object enter the scene. 
- `CaptureAllFromCamera()`: returns the next image and detections or classifications all together, given a camera name.

## Installation 
*in progress*

## Configure your `re-id-object-tracker` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `re-id-object-tracker` model. Enter a name for your service and click **Create**.

### Attributes description

The following attributes are available to configure your `re-id-object-tracker` module:




### TrackerConfig

| Name                      | Type   | Inclusion | Default       | Description                                                                                    |
| ------------------------- | ------ | --------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `lambda_value`            | float  | Optional  | `0.95`        | The lambda value is meant to adjust the contribution of the re-id matching and the IoU. The distance between to tracks equals: λ * feature_dist + (1 - λ) * (1 - IoU_score).|
| `max_age_track`           | int    | Optional  | `1e5`          | Maximum age (in frames) for a track to be considered active. Ranges from 0 to 1e5.             |
| `min_distance_threshold`  | float  | Optional  | `0.6`         | Minimum distance threshold for considering two features as distinct. Values range from 0 to 5. |
| `feature_distance_metric` | string | Optional  | `'euclidean'` | Metric used for calculating feature distance. Options include `cosine` and `euclidean`.        |
| `cooldown_period_s`       | float  | Optional  | `5`         | Duration for which the trigger `new_object_detected`.                                          |
| `start_fresh`       | bool  | Optional  | `False`         | Whether or|

### DetectorConfig

| Name                              | Type   | Inclusion    | Default | Description                                                                                                     |
| --------------------------------- | ------ | ------------ | ------- | --------------------------------------------------------------------------------------------------------------- |
| `detector_model_name`             | string | **Required** |         | Name of the model used for detection. Available options are `effDet0_int8`, `effDet0_fp16`, and `effDet0_fp32`. |
| `detection_threshold`             | float  | Optional     | `0.4`   | Confidence threshold for detecting objects, with values ranging from 0.0 to 1.0.                                |
| `detector_device`                 | string | Optional     |         | Device on which the detection model will run. Options are `cpu` and `gpu`.                                      |
| `detection_max_detection_results` | int    | Optional     | `5`     | Maximum number of detection results to return. Must be at least 1.                                              |

### FeatureEncoderConfig

| Name                      | Type   | Inclusion | Default | Description                                                                                                       |
| ------------------------- | ------ | --------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `feature_extractor_model` | string | Optional  |    `osnet_ain_x1_0`     | Name of the model used for feature extraction. Available options are `osnet_x0_25` and `osnet_ain_x1_0`. |
| `feature_encoder_device`  | string | Optional  |         | Device on which the feature encoder will run. Options are `cpu` and `gpu`.                                        |


### TracksManagerConfig

| Name                      | Type   | Inclusion    | Default | Description                                                                                     |
| ------------------------- | ------ | ------------ | ------- | ------------------------------------------------------------------------------------------------ |
| `path_to_database`        | string | **Required** |         | Path to the database where tracking information is stored.                                       |
| `save_period`             | int    | Optional     | `200`     | Frequency (in frames) at which the tracking information is saved to the database.               |


## `DoCommand()`

In this section, we provide example of `DoCommand` calls with the [Viam Python SDK](https://python.viam.dev/index.html).

### `relabel()`
The object tracker generate by default a unique ID string in the format  `"<category>_N_YYYYMMDD_HHMMSS"`. The object tracking module provide a way to relabel this default id.

```python
do_command_input = {
        "relabel": {"person_N_YYYYMMDD_HHMMSS": "known person"},
    }
do_command_res = await vision_object_tracker.do_command(do_command_input)
```

### `add()`

The `add()` command enables the user to pass a path to a directory containing pictures of people to be matched against the tracked people. If the distance between the embedding associated with a tracked object and the embedding computed from the pictures in the directory is smaller than the config attribute `re_id_threshold`, the label associated to the track is replaced by the `re_id_label` passed in the DoCommand `add`. Please note that the object tracking module gives priority to a label "manually" added with the `relabel()` command over a labeling resulting from a matching against embeddings added through the `add()` command.

```python
do_command_input = {"add": {re_id_label: path_to_directory}}
do_command_res = await vision_object_tracker.do_command(do_command_input)
```


### `delete()`

`delete()` deletes embeddings added with the `add()` command.
```python
do_command_input = {
        "delete": ["robin", "leon"]},
    }
do_command_res = await vision_object_tracker.do_command(do_command_input)
```









## Supplementaries

## PyInstaller build instructions
*in progress*
<!-- Run this to create your virtual environment:
```
./setup.sh
```

Run this to create your virtual environment:
Activate it bby running:
```
source .venv/bin/activate
```

Make sure that the requirements are installed:
```
pip3 install -r requirements.txt
```

Build the executable `dist/main`
```
python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data "./src/models/checkpoints:checkpoints"  src/main.py
``` -->
