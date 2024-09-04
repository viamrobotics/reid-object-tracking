from typing import Dict

import pytest
from fake_camera import FakeCamera
from google.protobuf.struct_pb2 import Struct
from viam.media.video import CameraMimeType
import logging
from viam.proto.app.robot import ServiceConfig
from viam.services.vision import Vision

from src.re_id_tracker import ReIDObjetcTracker
from src.tracker.tracker import Tracker
from src.utils import decode_image
from src.config.config import ReIDObjetcTrackerConfig
import os

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
    "lambda_value": 0.95,
    "max_age_track": 3000,
    "min_distance_threshold": 1.0,
    "feature_distance_metric": "euclidean",
    # DetectorConfig
    "detector_model_name": "effDet0_int8",
    "detection_threshold": 0.4,
    "detector_device": "cpu",
    "detection_max_detection_results": 5,
    # FeatureEncoderConfig
    "feature_extractor_model": "osnet_x0_25",
    "feature_encoder_device": "cpu",
}

IMG_PATH = "/Users/robin@viam.com/object-tracking/river_plate_2"

# IMG_PATH = "/path/to/something"


def get_config(config_dict: Dict) -> ServiceConfig:
    """returns a config populated with picture_directory and camera_name
    attributes.

    Returns:
        ServiceConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ServiceConfig(attributes=struct)
    return config


def get_vision_service(config_dict: Dict, reconfigure=True):
    service = ReIDObjetcTracker("test")
    cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)
    camera_name = cam.get_resource_name(CAMERA_NAME)
    cfg = get_config(config_dict)
    service.validate_config(cfg)
    if reconfigure:
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

    # @pytest.mark.asyncio
    # async def test_get_properties(self):
    #     service = ReIDObjetcTracker("test")
    #     p = await service.get_properties()
    #     assert p == PASSING_PROPERTIES

    # @pytest.mark.asyncio
    # async def test_get_detections_from_camera(self):
    #     service = get_vision_service(WORKING_CONFIG_DICT)
    #     for _ in range(10):
    #         get_detections_from_camera_result = (
    #             await service.get_detections_from_camera(
    #                 CAMERA_NAME, extra={}, timeout=0
    #             )
    #         )
    #     assert not get_detections_from_camera_result
    #     await service.close()

    @pytest.mark.asyncio
    async def test_tracker_logic(self):
        configure_logger()
        # service = get_vision_service(config_dict=WORKING_CONFIG_DICT, reconfigure=False)
        cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)

        cfg = ReIDObjetcTrackerConfig(get_config(WORKING_CONFIG_DICT))
        tracker = Tracker(cfg, cam, debug=True)
        for _ in range(48):
            viam_img = await cam.get_image(mime_type=CameraMimeType.JPEG)
            img = decode_image(viam_img)
            tracker.update(img)  # Update tracks

        await tracker.stop()


def configure_logger():
    logger = logging.getLogger("TrackDetectionLogger")
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to avoid duplication
    if not logger.handlers:
        # Ensure the log directory exists
        log_directory = "./results2"
        os.makedirs(log_directory, exist_ok=True)

        # Create a file handler
        file_handler = logging.FileHandler(
            os.path.join(log_directory, "cost_matrix.log")
        )
        file_handler.setLevel(logging.INFO)

        # Create a stream handler (optional)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Define a simple formatter
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
