import os
from typing import List

import numpy as np
from scipy.spatial.distance import euclidean
from torchreid.utils import FeatureExtractor

from src.config.config import FeatureEncoderConfig
from src.tracker.detector import Detection
from src.utils import resource_path

MODEL_NAME_TO_MODEL_RELATIVE_PATH = {
    "osnet_x0_25": os.path.join(
        "torchreid",
        "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    )
}


class FeatureEncoder:
    def __init__(self, cfg: FeatureEncoderConfig):
        """
        Initialize the FeatureEncoder with a feature extractor model.

        :param model_name: The name of the model to use for feature extraction.
        :param model_path: The path to the pre-trained model file.
        :param device: The device to run the model on ('cpu' or 'cuda').
        """

        model_name = cfg.feature_extractor_name.value
        model_path = MODEL_NAME_TO_MODEL_RELATIVE_PATH[model_name]

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=resource_path(model_path),
            device=cfg.device.value,
        )

    def compute_features(self, image: np.ndarray, detections: List[Detection]):
        """
        Compute features for each detection in the list.

        :param image: The original image as a numpy array (HxWxC).
        :param detections: A list of Detection objects.
        :return: A list of feature vectors for each detection.
        """
        # Extract bounding boxes from the detections
        cropped_images = [
            image[d.bbox[1] : d.bbox[3], d.bbox[0] : d.bbox[2]] for d in detections
        ]

        # TODO: need to decide if this should be here

        # # Preprocess cropped images for the feature extractor
        # processed_images = [
        #     cv2.cvtColor(
        #         cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB
        #     )  #  WE WILL SEE IF NEEDED
        #     for crop in cropped_images
        # ]

        # # Extract features
        # features = self.extractor(processed_images)

        features = self.extractor(cropped_images)

        return features

    def compute_distances(self, features):
        """
        Compute pairwise distances (Euclidean) between feature vectors.

        :param features: A list of feature vectors.
        :return: A dictionary with pairwise distances between the features.
        """
        distances = {}
        for i, feature in enumerate(features):
            for j, feature_2 in enumerate(features):
                if i != j:
                    dist = euclidean(np.asarray(feature), np.asarray(feature_2))
                    distances[f"feature_{i} - feature_{j}"] = dist
        return distances
