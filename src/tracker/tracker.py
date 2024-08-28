import datetime
from typing import Dict, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

from src.config.config import ReIDObjetcTrackerConfig
from src.tracker.detector import Detector
from src.tracker.encoder import FeatureEncoder
from viam.services.vision import Detection
from asyncio import Lock, Event, sleep, create_task
from viam.components.camera import CameraClient
from viam.media.video import CameraMimeType
from src.utils import decode_image
import asyncio
from viam.logging import getLogger

LOGGER = getLogger(__name__)


class Track:
    def __init__(self, track_id, bbox, feature_vector):
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

    def update(self, bbox, feature_vector):
        """
        Update the track with a new bounding box and feature vector.
        Also updates the velocity based on the difference between the last and current bbox.

        :param bbox: New bounding box coordinates.
        :param feature_vector: New feature vector.
        """
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

    def get_detection(self) -> Detection:
        return Detection(
            x_min=self.bbox[0],
            y_min=self.bbox[1],
            x_max=self.bbox[2],
            y_max=self.bbox[3],
            confidence=1,
            class_name=self.track_id,
        )


class Tracker:
    def __init__(self, cfg: ReIDObjetcTrackerConfig, camera):
        """
        Initialize the Tracker with a Detector for person detection and tracking logic.

        :param model_path: Path to the TFLite model file for the Detector.
        :param iou_threshold: Threshold for IoU matching.
        :param feature_threshold: Threshold for re-id feature matching.
        """

        # TODO: add mapping distance_name -> function

        self.camera: CameraClient = camera

        self.lambda_value = cfg.tracker_config.lambda_value.value
        self.distance_threshold = cfg.tracker_config.min_distance_threshold.value
        self.max_age_track = cfg.tracker_config.max_age_track.value
        self.distance = cfg.tracker_config.feature_distance_metric.value
        self.sleep_period = 1 / (cfg.tracker_config.max_frequency.value)

        self.detector = Detector(cfg.detector_config)
        self.encoder = FeatureEncoder(cfg.encoder_config)

        self.tracks: Dict[str, Track] = {}
        self.category_count: Dict[str, int] = {}

        self.current_tracks_id = set()
        self.current_detections = CurrentDetections()
        self.new_object_event = Event()
        self.new_object_notifier = NewObjectNotifier(
            self.new_object_event, cfg.tracker_config.cooldown_period.value
        )
        self.background_task = create_task(self._background_update_loop())
        self.stop_event = Event()

        self.color_map = {}
        self.colormap = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_COOL
        )[0]
        self.count = 0

    async def stop(self):
        """
        Stop the background loop by setting the stop event.
        """
        self.stop_event.set()
        await self.background_task  # Wait for the background task to finish

    async def _background_update_loop(self):
        """
        Background loop that continuously gets images from the camera and updates tracks.
        """
        while not self.stop_event.is_set():
            viam_img = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
            img = decode_image(viam_img)
            self.update(img)  # Update tracks
            await self.write_detections()  # Write detections to the shared object
            await sleep(self.sleep_period)

    async def write_detections(self):
        """
        Write the current detections from tracks into the shared CurrentDetections object.
        """
        cur_tracks = {
            track_id: self.tracks[track_id] for track_id in self.current_tracks_id
        }
        detections = [track.get_detection() for track in cur_tracks.values()]
        await self.current_detections.set_detections(detections)

    async def get_current_detections(self):
        """
        Get the current detections.
        """
        return await self.current_detections.get_detections()

    async def is_new_object_detected(self):
        return self.new_object_event.is_set()

    def update(self, img, visualize: bool = False):
        """
        Update the tracker with new detections.

        :param detections: List of Detection objects detected in the current frame.
        """

        # Get new detections
        detections = self.detector.detect(img)

        if not detections:
            self.current_tracks_id = set()
            for track in self.tracks.values():
                track.increment_age()
            return

        # Compute feature vectors for the current detections
        feature_vectors = self.encoder.compute_features(img, detections)

        # Initialize cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        # Create a list to keep track of updated track IDs
        updated_tracks_ids = []

        # New tracks
        new_tracks_ids = []

        # Calculate cost for each pair of track and detection
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = euclidean(track.feature_vector, feature_vectors[j])

                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                # TODO: figure out this 30
                cost_matrix[i, j] = self.lambda_value * feature_dist / 30 + (
                    1 - self.lambda_value
                ) * (1 - iou_score)

        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Keep track of the updated and unmatched tracks
        updated_tracks_ids = set()
        unmatched_detections = set(range(len(detections)))
        new_tracks_ids = set()

        # Update matched tracks
        for row, col in zip(row_indices, col_indices):
            if (
                cost_matrix[row, col] < self.distance_threshold
            ):  # Threshold to determine if a match is valid
                track_id = track_ids[row]
                detection = detections[col]
                self.tracks[track_id].update(detection.bbox, feature_vectors[col])
                updated_tracks_ids.add(track_id)
                unmatched_detections.discard(col)

        # Remove or age out tracks that were not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks_ids:
                self.tracks[track_id].increment_age()

                # Optionally remove old tracks
                if self.tracks[track_id].age > self.max_age_track:
                    del self.tracks[track_id]

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track_id = self.add_track(
                detections[detection_idx], feature_vectors[detection_idx]
            )
            new_tracks_ids.add(new_track_id)

        self.current_tracks_id = updated_tracks_ids.union(new_tracks_ids)

        # Set the new_object_event if new tracks were found
        if len(new_tracks_ids) > 0:
            self.new_object_notifier.notify_new_object()

        if visualize:
            bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for track_id in self.current_tracks_id:
                track = self.tracks[track_id]
                x1, y1, x2, y2 = track.bbox

                if track_id not in self.color_map:
                    self.color_map[track_id] = self.get_color_from_colormap(track_id)

                color = self.color_map[track_id]

                cv2.rectangle(
                    bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )
                # Put the score on the top-left corner of the bounding box
                cv2.putText(
                    bgr_image,
                    f"{track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )

            # Display the image with the detections
            cv2.imshow("Detections", bgr_image)
            cv2.imwrite(f"./results/res_{self.count}.png", bgr_image)
            self.count += 1
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

    def get_color_from_colormap(self, track_id):
        # Use the track_id to select a color from the colormap
        # Track_id modulo 256 to ensure it's within the range of the colormap
        track_id_int = hash(track_id) % 256

        # Use the hashed track_id to select a color from the colormap
        return tuple(int(c) for c in self.colormap[track_id_int])

    def add_track(self, detection, feature_vector):
        """
        Add a new track to the tracker using a Detection object.

        :param detection: Detection object containing bbox, score, and category.
        """
        # Generate a unique track ID
        track_id = self.generate_track_id(detection.category)

        # Create a new Track object and store it in the dictionary
        self.tracks[track_id] = Track(track_id, detection.bbox, feature_vector)
        return track_id

    def generate_track_id(self, category):
        """
        Generate a unique track ID based on the category and current date/time.

        :param category: The category of the detected object.
        :return: A unique track ID string in the format "<category>_N_YYYYMMDD_HHMMSS".
        """
        # Get the current count of this category
        if category not in self.category_count:
            self.category_count[category] = 0

        # Increment the count
        self.category_count[category] += 1
        count = self.category_count[category]

        # Get the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the track ID
        track_id = f"{category}_{count}_{timestamp}"

        return track_id

    def get_tracks(self):
        """
        Get the current list of active tracks.

        :return: List of active tracks.
        """
        return self.tracks


class CurrentDetections:
    def __init__(self):
        self.detections: List[Detection] = []
        self.lock = Lock()

    async def get_detections(self):
        """
        Safely get the current detections.
        """
        async with self.lock:
            return list(self.detections)

    async def set_detections(self, new_detections: List[Detection]):
        """
        Safely set the current detections.
        """
        async with self.lock:
            self.detections = new_detections


class NewObjectNotifier:
    def __init__(self, new_object_event: Event, cooldown_period_s: int):
        """
        Initialize the notifier with a cooldown period.

        :param cooldown_seconds: Time in seconds for the cooldown period.
        """
        self.cooldown_seconds = cooldown_period_s
        self.new_object_event = new_object_event
        self.cooldown_task = None

    def notify_new_object(self):
        """
        Notify that a new object has been detected and restart the cooldown.
        """
        # Set the event to notify about the new object
        self.new_object_event.set()

        # Cancel any existing cooldown task
        if self.cooldown_task is not None:
            self.cooldown_task.cancel()

        # Start a new cooldown task
        self.cooldown_task = asyncio.create_task(self._clear_event_after_cooldown())

    async def _clear_event_after_cooldown(self):
        """
        Clear the event after the cooldown period.
        """
        try:
            await asyncio.sleep(self.cooldown_seconds)
            self.new_object_event.clear()
        except asyncio.CancelledError:
            pass
