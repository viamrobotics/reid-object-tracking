# VIAM RE-ID OBJECT TRACKER

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for tracking object using ReID.


![example](img/output.gif)

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:re-id-object-tracker` model from the [`re-id-object-tracker` module](https://app.viam.com/module/re-id-object-tracker).
<!-- TODO: check link -->
This module implements the following methods of the [vision service API](https://docs.viam.com/services/vision/#api).


GetDetections(): returns the bounding boxes with the unique id as label and the object detection confidence as confidence.

GetClassifications(): an image will be classified with the label `new_object_detected` when a new object enter the scene. 

CaptureAllFromCamera()

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
| `lambda_value`            | float  | Optional  | `0.95`        | A value used in tracking computations, ranging between 0 and 1.                                |
| `max_age_track`           | int    | Optional  | `30`          | Maximum age (in frames) for a track to be considered active. Ranges from 0 to 100.             |
| `min_distance_threshold`  | float  | Optional  | `1.0`         | Minimum distance threshold for considering two features as distinct. Values range from 0 to 5. |
| `feature_distance_metric` | string | Optional  | `'euclidean'` | Metric used for calculating feature distance. Options include `cosine` and `euclidean`.        |
| `cooldown_period_s`       | float  | Optional  | `'5'`         | Duration for which the trigger `new_object_detected`.                                          |

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
| `feature_extractor_model` | string | Optional  |         | Name of the model used for feature extraction. Available options are `osnet_x0_25`, `osnet_x0_5`, and `osnet_x1`. |
| `feature_encoder_device`  | string | Optional  |         | Device on which the feature encoder will run. Options are `cpu` and `gpu`.                                        |


### TracksManagerConfig

| Name                      | Type   | Inclusion    | Default | Description                                                                                     |
| ------------------------- | ------ | ------------ | ------- | ------------------------------------------------------------------------------------------------ |
| `path_to_database`        | string | **Required** |         | Path to the database where tracking information is stored.                                       |
| `save_period`             | int    | Optional     | `5`     | Frequency (in seconds) at which the tracking information is saved to the database.               |









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
