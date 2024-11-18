import os
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cityblock, cosine, euclidean
from torchreid.utils import FeatureExtractor

from src.config.config import FeatureEncoderConfig
from src.tracker.detector.detection import Detection
from src.utils import resource_path


class EncoderModelConfig:
    def __init__(
        self,
        model_file: str,
        repository: str,
        metric: str,
        mean: float = 0,
        std: float = 1,
    ):
        self.model_file = model_file
        self.repository = repository
        self.metric = metric
        self.mean = mean
        self.std = std

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join(self.repository, self.model_file)
        return resource_path(relative_path)


ENCODERS_CONFIG: Dict[str, EncoderModelConfig] = {
    "osnet_x0_25": EncoderModelConfig(
        model_file="osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        repository="torchreid",
        metric="euclidean",
        mean=10,
        std=(35 - 10),
    ),
    "osnet_ain_x1_0": EncoderModelConfig(
        model_file="osnet_ain_ms_d_c.pth.tar",
        repository="torchreid",
        metric="cosine",
        mean=0,
        std=1,
    ),
    "model_2": None,
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
        self.model_config = ENCODERS_CONFIG.get(model_name)

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=self.model_config.get_model_path(),
            device=cfg.device.value,
        )

    def compute_features(
        self, image: np.ndarray, detections: List[Detection]
    ) -> List[np.ndarray]:
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

        features = self.extractor(cropped_images)
        return [np.array(feature, dtype=np.float32) for feature in features]

    def compute_distance(self, feature_vector_1, feature_vector_2):
        """
        Compute pairwise distances (Euclidean) between feature vectors.

        :param features: A list of feature vectors.
        :return: A dictionary with pairwise distances between the features.
        """

        if self.model_config.metric == "euclidean":
            distance = euclidean(feature_vector_1, feature_vector_2)
        elif self.model_config.metric == "cosine":
            distance = cosine(feature_vector_1, feature_vector_2)
        elif self.model_config.metric == "manhattan":
            distance = cityblock(feature_vector_1, feature_vector_2)
        else:
            raise ValueError(f"Unsupported metric '{self.model_config.metric }'")

        return (distance - self.model_config.mean) / self.model_config.std
