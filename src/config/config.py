from viam.proto.app.robot import ServiceConfig

from src.config.attribute import FloatAttribute, IntAttribute, StringAttribute


class TrackerConfig:
    def __init__(self, config: "ServiceConfig"):
        self.lambda_value = FloatAttribute(
            field_name="lambda_value",
            config=config,
            min_value=0,
            max_value=1,
            default_value=0.95,
        )
        self.max_age_track = IntAttribute(
            field_name="max_age_track",
            config=config,
            min_value=0,
            max_value=10e5,
            default_value=30,
        )
        self.min_distance_threshold = FloatAttribute(
            field_name="min_distance_threshold",
            config=config,
            min_value=0,
            max_value=2,
            default_value=1.0,
        )
        self.feature_distance_metric = StringAttribute(
            field_name="feature_distance_metric",
            config=config,
            default_value="euclidean",
            allowlist=["cosine", "euclidean"],
        )

        self.max_frequency = FloatAttribute(
            field_name="max_frequency_hz",
            config=config,
            default_value=30,
            min_value=1,
            max_value=100,
        )

        self.cooldown_period = FloatAttribute(
            field_name="cooldown_period_s",
            config=config,
            default_value=5,
            min_value=0,
        )


class DetectorConfig:
    def __init__(self, config: "ServiceConfig"):
        self.model_name = StringAttribute(
            field_name="detector_model_name",
            config=config,
            default_value="effDet0_int8",
            allowlist=["effDet0_int8", "effDet0_fp16", "effDet0_fp32"],
        )
        self.threshold = FloatAttribute(
            field_name="detection_threshold",
            config=config,
            min_value=0.0,
            max_value=1.0,
            default_value=0.4,
        )
        self.device = StringAttribute(
            field_name="detector_device",
            config=config,
            default_value="cpu",
            allowlist=["cpu", "gpu"],
        )

        self.max_results = IntAttribute(
            field_name="detection_max_detection_results",
            config=config,
            default_value=5,
            min_value=1,
        )

        # TODO: add a filter for detection label and confidence here


class FeatureEncoderConfig:
    def __init__(self, config: "ServiceConfig"):
        self.feature_extractor_name = StringAttribute(
            field_name="feature_extractor_model",
            config=config,
            default_value="osnet_x0_25",
            allowlist=["osnet_x0_25", "osnet_x0_5", "osnet_x1"],
        )

        self.device = StringAttribute(
            field_name="feature_encoder_device",
            config=config,
            default_value="cpu",
            allowlist=["cpu", "gpu"],
        )


class ReIDObjetcTrackerConfig:
    def __init__(self, config: ServiceConfig):
        self.config = config

        self.tracker_config = TrackerConfig(config)
        self.detector_config = DetectorConfig(config)
        self.encoder_config = FeatureEncoderConfig(config)
