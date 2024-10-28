# VIAM RE-ID OBJECT TRACKER

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for tracking object using ReID.


![example](img/output.gif)

## Getting started

This module implements the following methods of the [vision service API](https://docs.viam.com/services/vision/#api).

GetDetections(): returns the bounding boxes with the unique id as label and the object detection confidence as confidence.

GetClassifications(): an image will be classified with the label `new_object_detected` when a new object enters the scene. 

CaptureAllFromCamera()

## Installation 
*in progress*

## Configure your `re-id-object-tracker` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the [**CONFIGURE** tab](https://docs.viam.com/configure/) of your [machine](https://docs.viam.com/fleet/machines/) in the [Viam app](https://app.viam.com/).
[Add vision / re-id-object-tracker to your machine](https://docs.viam.com/configure/#components).

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

In addition to the [vision service API](https://docs.viam.com/services/vision/#api), the `re-id-object-tracking` module supports some model-specific commands that allow you to add, delete, relabel and list people.
You can invoke these commands by passing appropriately keyed JSON documents to the [`DoCommand()`](/appendix/apis/services/vision/#docommand) method using one of [Viam's SDKs](https://docs.viam.com/sdks/).

For example:

```json
{
  "add": {
    "amanda": ["/path/to/amanda-pictures/dir"], 
    "bob": ["/path/to/bob-pictures/dir"]
  }
}
```

```json
{
  "delete": ["amanda","bob"]
}
```

```json
{
  "relabel": {
    "person1_30122024": "amanda",
    "person2_20112024": "bob"
  }
}
```

```json
{
  "list": "true"
}
```

| Key | Description |
| ----- | --------- |
| `add` | Add a new person or people to the authorized label list. Use a directory with pictures of the person from that day. Re-ID will not work with different clothes. |
| `delete` | Delete a new person or people from the authorized label list by label. |
| `relabel` | Relabel a person or people by ID. |
| `list` | List known people's labels, IDs and whether they are authorized. |

If the distance between the embedding associated with a tracked object and the embedding computed from the pictures in the directory is smaller than the config attribute `re_id_threshold`, the label associated to the track is replaced by the `re_id_label` passed in the DoCommand `add`.
Please note that the object tracking module gives priority to a label "manually" added with the `relabel()` command over a labeling resulting from a matching against embeddings added through the `add()` command.

For more information, see [`DoCommand()`](/appendix/apis/services/vision/#docommand).

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
