import numpy as np
from viam.services.vision import Detection
from asyncio import Lock


class Track:
    def __init__(self, track_id, bbox, feature_vector, distance, label=None):
        """
        Initialize a track with a unique ID, bounding box, and re-id feature vector.

        :param track_id: Unique identifier for the track.
        :param bbox: Bounding box coordinates [x1, y1, x2, y2].
        :param feature_vector: Feature vector for re-id matching.
        """
        self.track_id = track_id
        self.bbox = np.array(bbox)
        self.predicted_bbox = np.array(bbox)
        self.feature_vector = np.array(feature_vector)
        self.age = 0  # Time since the last update
        self.history = [np.array(bbox)]  # Stores past bounding boxes for this track
        self.velocity = np.array([0, 0, 0, 0])  # Initial velocity (no motion)
        self.distance = distance

        self.label = label
        self.label_lock = Lock()
        self.label_from_reid = None

    def __eq__(self, other) -> bool:
        """
        To test serialization, so just on unique id, bbox, feature vector
        """
        if not isinstance(other, Track):
            return False
        return (
            self.track_id == other.track_id
            and np.array_equal(self.bbox, other.bbox)
            and np.array_equal(self.feature_vector, other.feature_vector)
        )

    def update(self, bbox, feature_vector, distance):
        """
        Update the track with a new bounding box and feature vector.
        Also updates the velocity based on the difference between the last and current bbox.

        :param bbox: New bounding box coordinates.
        :param feature_vector: New feature vector.
        """
        self.distance = distance
        bbox = np.array(bbox)
        self.velocity = bbox - self.bbox  # Update velocity
        self.bbox = bbox
        self.feature_vector = np.array(feature_vector)
        self.age = 0
        self.history.append(bbox)
        self.predicted_bbox = self.predict()

    def predict(self):
        """
        Predict the next position based on the current velocity and last known position.

        :return: Predicted bounding box coordinates.
        """
        predicted_bbox = self.bbox + self.velocity
        return predicted_bbox

    def increment_age(self):
        """
        Increment the age and time since update for the track.
        If not updated, the prediction is updated using the velocity.
        """
        self.age += 1
        self.bbox = self.predict()  # Update the bbox with the predicted one

    def iou(self, bbox):
        """
        Calculate Intersection over Union (IoU) between this track's bbox and another bbox.

        :param bbox: Bounding box to compare with.

            (x1, y1) --------------+
                |                  |
                |                  |
                |                  |
                +-------------- (x2, y2)

        :return: IoU score.
        """
        x1_t, y1_t, x2_t, y2_t = self.predicted_bbox
        x1_o, y1_o, x2_o, y2_o = bbox

        # Determine the coordinates of the intersection rectangle
        x1_inter = max(x1_t, x1_o)
        y1_inter = max(y1_t, y1_o)
        x2_inter = min(x2_t, x2_o)
        y2_inter = min(y2_t, y2_o)

        # Compute the area of intersection
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # Compute the area of both the prediction and ground-truth rectangles
        track_area = (x2_t - x1_t) * (y2_t - y1_t)
        other_area = (x2_o - x1_o) * (y2_o - y1_o)

        # Compute the Intersection over Union (IoU)
        union_area = track_area + other_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def feature_distance(self, feature_vector):
        """
        Calculate the distance between this track's feature vector and another feature vector.

        :param feature_vector: Feature vector to compare with.
        :return: Distance (e.g., Euclidean distance).
        """
        return np.linalg.norm(self.feature_vector - feature_vector)

    async def get_detection(self) -> Detection:
        label = await self.get_label()
        return Detection(
            x_min=self.bbox[0],
            y_min=self.bbox[1],
            x_max=self.bbox[2],
            y_max=self.bbox[3],
            confidence=1 - self.distance,
            class_name=label,
        )

    def serialize(self):
        return self.track_id, self.bbox.tobytes(), self.feature_vector.tobytes()

    def add_label(self, label_from_reid):
        self.label_from_reid = label_from_reid

    def get_embedding(self):
        return self.feature_vector

    async def relabel(self, new_label):
        async with self.label_lock:
            self.label = new_label

    def relabel_reid_label(self, label: str):
        self.label_from_reid = label

    async def get_label(self):
        """
        Tries to return label first, then label_from_reid
        and finally track_id
        """
        async with self.label_lock:
            label = self.label
            if label is not None:
                return label
        if self.label_from_reid is not None:
            return self.label_from_reid
        return self.track_id

    def _get_label(self):
        if self.label is not None:
            return self.label
        if self.label_from_reid is not None:
            return self.label_from_reid
        return self.track_id
