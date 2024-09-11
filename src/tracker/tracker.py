import asyncio
import datetime
from asyncio import Event, Lock, create_task, sleep
from typing import Dict, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from viam.components.camera import CameraClient
from viam.logging import getLogger
from viam.media.video import CameraMimeType
from viam.services.vision import Detection

from src.config.config import ReIDObjetcTrackerConfig
from src.tracker.detector import Detector
from src.tracker.encoder import FeatureEncoder
from src.tracker.tracks_manager import TracksManager
from src.tracker.track import Track
from src.utils import decode_image, log_tracks_info, log_cost_matrix

LOGGER = getLogger(__name__)


class Tracker:
    def __init__(self, cfg: ReIDObjetcTrackerConfig, camera, debug: bool = False):
        """
        Initialize the Tracker with a Detector for person detection and tracking logic.

        :param model_path: Path to the TFLite model file for the Detector.
        :param iou_threshold: Threshold for IoU matching.
        :param feature_threshold: Threshold for re-id feature matching.
        """
        self.camera: CameraClient = camera

        self.lambda_value = cfg.tracker_config.lambda_value.value
        self.distance_threshold = cfg.tracker_config.min_distance_threshold.value
        self.max_age_track = cfg.tracker_config.max_age_track.value
        self.distance = cfg.tracker_config.feature_distance_metric.value
        self.sleep_period = 1 / (cfg.tracker_config.max_frequency.value)

        self.detector = Detector(cfg.detector_config)
        self.encoder = FeatureEncoder(cfg.encoder_config)

        self.tracks: Dict[str, Track] = {}
        self.tracks_manager = TracksManager(cfg.tracks_manager_config)
        self.start_fresh: bool = True  # TODO should be an option in the config
        self.category_count: Dict[str, int] = {}

        self.current_tracks_id = set()
        self.background_task = None
        self.current_detections = CurrentDetections()
        self.new_object_event = Event()
        self.new_object_notifier = NewObjectNotifier(
            self.new_object_event, cfg.tracker_config.cooldown_period.value
        )
        self.stop_event = Event()

        self.debug = debug
        self.color_map = {}
        self.colormap = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_COOL
        )[0]
        self.count = 0

    def start(self):
        if not self.start_fresh:
            # Parse tracks on database
            self.tracks_manager.parse_tracks_on_disk()
            self.import_tracks_from_tracks_manager()

            # Parse category_count on database
            self.tracks_manager.parse_category_count_on_disk()
            self.import_category_count_from_tracks_manager()

        # start
        self.background_task = create_task(self._background_update_loop())

    async def stop(self):
        """
        Stop the background loop by setting the stop event.
        """
        self.stop_event.set()
        self.new_object_notifier.close()
        if self.background_task is not None:
            await self.background_task  # Wait for the background task to finish
        self.tracks_manager.close()

    def import_tracks_from_tracks_manager(self):
        self.tracks = self.tracks_manager.get_tracks_on_disk()

    def import_category_count_from_tracks_manager(self):
        self.category_count = self.tracks_manager.get_category_count_on_disk()

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

        # Keep track of the old tracks, updated and unmatched tracks
        all_old_tracks_id = set(self.tracks.keys())
        updated_tracks_ids = set()
        unmatched_detections = set(range(len(detections)))
        new_tracks_ids = set()

        if not detections:
            self.current_tracks_id = set()
            self.increment_age_and_delete_tracks()
            if self.debug:
                log_tracks_info(
                    updated_tracks_ids=updated_tracks_ids,
                    new_tracks_ids=new_tracks_ids,
                    lost_tracks_ids=all_old_tracks_id - updated_tracks_ids,
                )
            return

        # Compute feature vectors for the current detections
        feature_vectors = self.encoder.compute_features(img, detections)

        # Initialize cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        # Calculate cost for each pair of track and detection
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = self.encoder.compute_distance(
                    track.feature_vector, feature_vectors[j]
                )
                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    1 - self.lambda_value
                ) * (1 - iou_score)

                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    0.1
                ) * (  # TODO: test to find more robust parameters
                    1 - iou_score
                )
        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        # Update matched tracks
        for row, col in zip(row_indices, col_indices):
            distance = cost_matrix[row, col]
            if (
                distance < self.distance_threshold
            ):  # Threshold to determine if a match is valid
                track_id = track_ids[row]
                detection = detections[col]
                self.tracks[track_id].update(
                    bbox=detection.bbox,
                    feature_vector=feature_vectors[col],
                    distance=distance,
                )
                updated_tracks_ids.add(track_id)
                unmatched_detections.discard(col)

        self.increment_age_and_delete_tracks(updated_tracks_ids)

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track_id = self.add_track(
                detection=detections[detection_idx],
                feature_vector=feature_vectors[detection_idx],
            )
            new_tracks_ids.add(new_track_id)

        self.current_tracks_id = updated_tracks_ids.union(new_tracks_ids)

        # Set the new_object_event if new tracks were found
        if len(new_tracks_ids) > 0:
            self.new_object_notifier.notify_new_object()

        self.count += 1

        # Write only the updated tracks on the db
        # TODO: only pass updated tracks so we don't serialize everytime
        if self.count % self.tracks_manager.save_period.value == 0:
            self.tracks_manager.write_tracks_on_db(self.tracks)
            self.tracks_manager.write_category_count_on_db(self.category_count)

        if self.debug:
            log_tracks_info(
                updated_tracks_ids=updated_tracks_ids,
                new_tracks_ids=new_tracks_ids,
                lost_tracks_ids=all_old_tracks_id - updated_tracks_ids,
            )
            log_cost_matrix(
                cost_matrix=cost_matrix,
                track_ids=list(self.tracks.keys()),
                iteration_number=self.count,
            )

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

            cv2.imwrite(f"./results2/result_{self.count}.png", bgr_image)

            # cv2.imshow("Detections", bgr_image)
            # cv2.waitKey(0)  # Wait for a key press to close the window
            # cv2.destroyAllWindows()

    def get_color_from_colormap(self, track_id):
        # Use the track_id to select a color from the colormap
        # Track_id modulo 256 to ensure it's within the range of the colormap
        track_id_int = hash(track_id) % 256

        # Use the hashed track_id to select a color from the colormap
        return tuple(int(c) for c in self.colormap[track_id_int])

    def add_track(self, detection, feature_vector, distance=0):
        """
        Add a new track to the tracker using a Detection object.

        :param detection: Detection object containing bbox, score, and category.
        """
        # Generate a unique track ID
        track_id = self.generate_track_id(detection.category)

        # Create a new Track object and store it in the dictionary
        self.tracks[track_id] = Track(
            track_id=track_id,
            bbox=detection.bbox,
            feature_vector=feature_vector,
            distance=distance,
        )
        return track_id

    def increment_age_and_delete_tracks(self, updated_tracks_ids=[]):
        # Remove or age out tracks that were not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks_ids:
                self.tracks[track_id].increment_age()

                # Optionally remove old tracks
                if self.tracks[track_id].age > self.max_age_track:
                    del self.tracks[track_id]

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

    def close(self):
        self.new_object_event.clear()
        if self.cooldown_task is not None:
            self.cooldown_task.cancel()

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
