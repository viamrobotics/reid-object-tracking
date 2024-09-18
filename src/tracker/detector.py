import os
from typing import List, Dict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from src.config.config import DetectorConfig
from src.utils import resource_path


class DetectorModelConfig:
    def __init__(self, path: str):
        self.path = path

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join("mediapipe", self.path)
        return resource_path(relative_path)


DETECTORS_CONFIG: Dict[str, DetectorModelConfig] = {
    "effDet0_int8": DetectorModelConfig(
        path="efficientdet_int8.tflite",
    ),
    "model_2": None,
}


class Detection:
    def __init__(self, bbox, score, category):
        """
        Initialize a detection with a bounding box, confidence score, and category.

        :param bbox: Bounding box [x1, y1, x2, y2].
        :param score: Confidence score for the detection.
        :param category: Category of the detected object as a string.
        """
        self.bbox = bbox
        self.score = score
        self.category = category

    def iou(self, other):
        """
        Calculate the Intersection over Union (IoU) between this detection's bounding box
        and another detection's bounding box.

        :param other: Another Detection instance.
        :return: IoU as a float value.
        """
        x1_self, y1_self, x2_self, y2_self = self.bbox
        x1_other, y1_other, x2_other, y2_other = other.bbox

        # Determine the coordinates of the intersection rectangle
        x1_inter = max(x1_self, x1_other)
        y1_inter = max(y1_self, y1_other)
        x2_inter = min(x2_self, x2_other)
        y2_inter = min(y2_self, y2_other)

        # Compute the area of intersection
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # Compute the area of both bounding boxes
        self_area = (x2_self - x1_self) * (y2_self - y1_self)
        other_area = (x2_other - x1_other) * (y2_other - y1_other)

        # Compute the Intersection over Union (IoU)
        union_area = self_area + other_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def __repr__(self):
        """
        Return a string representation of the detection for easy printing.
        """
        return f"Detection(bbox={self.bbox}, score={self.score:.2f}, category='{self.category}')"


class Detector:
    def __init__(self, cfg: DetectorConfig):
        """
        Initialize the Detector with a given model and options.

        :param model_path: Path to the TFLite model file.
        :param score_threshold: Minimum confidence score for detections.
        :param max_results: Maximum number of detection results to return.
        """

        model_name = cfg.model_name.value
        self.model_config = DETECTORS_CONFIG.get(model_name)

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_config.get_model_path()
            ),
            max_results=cfg.max_results.value,
            running_mode=VisionRunningMode.IMAGE,
            score_threshold=cfg.threshold.value,
            category_allowlist=["person"],
        )
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def detect(self, image: np.ndarray, visualize: bool = False) -> List[Detection]:
        """
        Detect persons in the given image and return bounding boxes and scores.

        :param image: Image provided as a NumPy array.
        :return: List of tuples, where each tuple contains a bounding box [x1, y1, x2, y2] and a confidence score.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)

        detections = []
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            score = detection.categories[0].score  # category "0" is person
            category = detection.categories[0].category_name

            detections.append(
                (
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        category=category,
                    )
                )
            )

            if visualize:
                cv2.rectangle(
                    bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                # Put the score on the top-left corner of the bounding box
                cv2.putText(
                    bgr_image,
                    f"{category}: {score:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if visualize:
            cv2.imshow("Detections", bgr_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections
