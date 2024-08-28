from typing import Dict

import pytest
from fake_camera import FakeCamera
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig
from viam.services.vision import Vision

from src.re_id_tracker import ReIDObjetcTracker

CAMERA_NAME = "fake-camera"

PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=True,
    detections_supported=True,
    object_point_clouds_supported=False,
)

MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "camera_name": CAMERA_NAME,
    # TrackerConfig
    "lambda_value": 0.85,
    "max_age_track": 30,
    "min_distance_threshold": 1.1,
    "feature_distance_metric": "euclidean",
    # DetectorConfig
    "detector_model_name": "effDet0_int8",
    "detection_threshold": 0.4,
    "detector_device": "gpu",
    "detection_max_detection_results": 5,
    # FeatureEncoderConfig
    "feature_extractor_model": "osnet_x0_25",
    "feature_encoder_device": "cpu",
}

IMG_PATH = "/Users/robinin/object_tracking/river_plate_2"

# IMG_PATH = "/path/to/something"


def get_config(config_dict: Dict):
    """returns a config populated with picture_directory and camera_name
    attributes.

    Returns:
        ServiceConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ServiceConfig(attributes=struct)
    return config


def get_vision_service(config_dict: Dict):
    service = ReIDObjetcTracker("test")
    cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)
    camera_name = cam.get_resource_name(CAMERA_NAME)
    cfg = get_config(config_dict)
    service.validate_config(cfg)
    service.reconfigure(cfg, dependencies={camera_name: cam})
    return service


class TestFaceReId:
    # def test_config(self):
    #     with pytest.raises(ValueError):
    #         service = get_vision_service(config_dict={})

    # def test_wrong_encoder_config(self):
    #     cfg = WORKING_CONFIG_DICT.copy()
    #     cfg["extractor_model"] = "not-a-real-model-name"
    #     with pytest.raises(ValueError):
    #         _ = get_vision_service(cfg)

    @pytest.mark.asyncio
    async def test_get_properties(self):
        service = ReIDObjetcTracker("test")
        p = await service.get_properties()
        assert p == PASSING_PROPERTIES

    @pytest.mark.asyncio
    async def test_get_detections_from_camera(self):
        service = get_vision_service(WORKING_CONFIG_DICT)
        for _ in range(10):
            get_detections_from_camera_result = (
                await service.get_detections_from_camera(
                    CAMERA_NAME, extra={}, timeout=0
                )
            )
        assert not get_detections_from_camera_result
        await service.close()
