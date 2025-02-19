import asyncio
import datetime
import os
from asyncio import Event, create_task, sleep
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from viam.components.camera import CameraClient
from viam.logging import getLogger
from viam.media.video import CameraMimeType
from viam.proto.service.vision import Detection

from src.config.config import ReIDObjectTrackerConfig
from src.image.image import ImageObject
from src.tracker.detector.detector import Detector, get_detector
from src.tracker.encoder.feature_encoder import FeatureEncoder, get_encoder
from src.tracker.face_id.identifier import FaceIdentifier
from src.tracker.track import Track
from src.tracker.tracks_manager import TracksManager
from src.utils import log_cost_matrix, log_tracks_info


LOGGER = getLogger(__name__)


class NoPersonDetectedError(Exception):
    pass


class Tracker:
    def __init__(self, cfg: ReIDObjectTrackerConfig, camera, debug: bool = False):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector: Detector = get_detector(cfg.detector_config)
        self.encoder: FeatureEncoder = get_encoder(cfg.encoder_config)
        self.face_identifier: FaceIdentifier = FaceIdentifier(cfg.face_id_config)

        self.tracks: Dict[str, Track] = {}

        self.track_candidates: List[Track] = []
        self.tracks_manager = TracksManager(cfg.tracks_manager_config)
        self.start_fresh: bool = cfg.tracker_config.start_fresh.value
        self.save_to_db: bool = cfg.tracker_config.save_to_db.value

        self.minimum_track_persistance: int = (
            cfg.tracker_config.min_track_persistence.value
        )

        self.category_count: Dict[str, int] = {}

        self.track_ids_with_label: Dict[str, List[str]] = {}

        self.labeled_person_embeddings: Dict[str, List[torch.Tensor]] = {}
        self.path_to_known_persons = cfg.tracker_config.path_to_known_persons.value

        self.reid_threshold = cfg.tracker_config.re_id_threshold.value

        self.current_tracks_id = set()
        self.background_task = None
        self.new_object_event = Event()
        self.new_object_notifier = NewObjectNotifier(
            self.new_object_event, cfg.tracker_config.cooldown_period.value
        )
        self.stop_event = Event()

        self.debug = debug
        self.count = 0

        self.start_background_loop = cfg.tracker_config._start_background_loop

        self.compute_known_persons_embeddings()

    def start(self):
        if not self.start_fresh:
            # Parse tracks on database
            self.tracks_manager.parse_tracks_on_disk()
            self.import_tracks_from_tracks_manager()

            # Parse category_count on database
            self.tracks_manager.parse_category_count_on_disk()
            self.import_category_count_from_tracks_manager()

            self.tracks_manager.parse_map_label_track_ids()
            self.import_track_ids_with_label_from_tracks_manager()

            for label, track_ids in self.track_ids_with_label.items():
                for track_id in track_ids:
                    if track_id not in self.tracks:
                        # TODO: log an error
                        continue
                    self.tracks[track_id].add_label(label)

        # start
        if self.start_background_loop:
            self.background_task = create_task(self._background_update_loop())

    async def stop(self):
        """
        Stop the background loop by setting the stop event.
        """
        self.stop_event.set()
        self.new_object_notifier.close()
        self.tracks_manager.close()
        if self.background_task is not None:
            await self.background_task  # Wait for the background task to finish

    def import_tracks_from_tracks_manager(self):
        self.tracks = self.tracks_manager.get_tracks_on_disk()

    def import_category_count_from_tracks_manager(self):
        self.category_count = self.tracks_manager.get_category_count_on_disk()

    def import_track_ids_with_label_from_tracks_manager(self):
        self.track_ids_with_label = (
            self.tracks_manager.get_track_ids_with_label_on_disk()
        )

    async def _background_update_loop(self):
        """
        Background loop that continuously gets images from the camera and updates tracks.
        """
        while not self.stop_event.is_set():
            img = await self.get_and_decode_img()
            # TODO: check img here
            if img is not None:
                self.update(img)  # Update tracks
            await sleep(self.sleep_period)

    async def get_and_decode_img(self):
        try:
            viam_img = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        except:
            return None
        return ImageObject(viam_img, self.device)

    def relabel_tracks(self, dict_old_label_new_label: Dict[str, str]):
        answer = {}
        for track_id, new_label in dict_old_label_new_label.items():
            if track_id not in self.tracks:
                answer[track_id] = (
                    f"DoCommand relabelling error: couldn't find tracks with the ID: {track_id}"
                )
                continue

            self.tracks[track_id].label = new_label
            answer[track_id] = f"success: changed label to '{new_label}'"
        return answer

    def get_current_detections(self):
        """
        Get the current detections.
        """
        dets = []

        for track in self.tracks.values():
            if track.is_detected():
                dets.append(track.get_detection())

        for track in self.track_candidates:
            if track.is_detected():
                dets.append(track.get_detection(self.minimum_track_persistance))
        return dets

    async def is_new_object_detected(self):
        return self.new_object_event.is_set()

    def update(
        self,
        img: ImageObject,
    ):
        """
        Update the tracker with new detections.

        :param detections: List of Detection objects detected in the current frame.
        """
        self.clear_detected_track()
        # Get new detections
        detections = self.detector.detect(img)

        # Keep track of the old tracks, updated and unmatched tracks
        all_old_tracks_id = set(self.tracks.keys())
        updated_tracks_ids = set()
        unmatched_detections = set(range(len(detections)))
        new_tracks_ids = set()
        current_track_candidates = []

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
        features_vectors = self.encoder.compute_features(img, detections)

        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices, cost_matrix = self.get_matching_tracks(
            tracks=self.tracks, detections=detections, feature_vectors=features_vectors
        )
        # Update matched tracks
        for row, col in zip(row_indices, col_indices):
            distance = cost_matrix[row, col]
            if (
                distance < self.distance_threshold
            ):  # Threshold to determine if a match is valid
                track_id = list(self.tracks.keys())[row]
                track = self.tracks[track_id]
                detection = detections[col]
                self.tracks[track_id].update(
                    bbox=detection.bbox,
                    feature_vector=features_vectors[col],
                    distance=distance,
                )
                track.set_is_detected()
                updated_tracks_ids.add(track_id)
                unmatched_detections.discard(col)

        # Find match with track candidate
        if len(unmatched_detections) > 0:
            if len(self.track_candidates) < 1:
                for detection_id in unmatched_detections:
                    detection = detections[detection_id]
                    feature_vector = features_vectors[detection_id]

                    self.add_track_candidate(
                        detection=detection,
                        feature_vector=feature_vector,
                    )
                    current_track_candidates.append(len(self.track_candidates) - 1)
            else:
                track_candidate_idx, unmatched_detection_idx, cost_matrix = (
                    self.get_matching_track_candidates(
                        detections=[detections[i] for i in unmatched_detections],
                        features_vectors=features_vectors,
                    )
                )
                promoted_track_candidates = []
                for track_candidate_id, unmatched_detection_id in zip(
                    track_candidate_idx, unmatched_detection_idx
                ):
                    distance = cost_matrix[track_candidate_id, unmatched_detection_id]
                    detection = detections[unmatched_detection_id]
                    matching_track_candidate = self.track_candidates[track_candidate_id]

                    if distance < self.distance_threshold:
                        matching_track_candidate.update(
                            bbox=detection.bbox,
                            feature_vector=features_vectors[unmatched_detection_id],
                            distance=distance,
                        )
                        matching_track_candidate.increment_persistence()
                        if (
                            matching_track_candidate.get_persistence()
                            > self.minimum_track_persistance
                        ):
                            promoted_track_candidates.append(track_candidate_id)
                            new_track_id = self.promote_to_track(
                                track_candidate_id,
                                feature_vector=features_vectors[unmatched_detection_id],
                            )
                            new_tracks_ids.add(new_track_id)
                            self.tracks[new_track_id].set_is_detected()
                        else:
                            matching_track_candidate.set_is_detected()

                    else:
                        self.add_track_candidate(
                            detection=detection,
                            feature_vector=features_vectors[unmatched_detection_id],
                        )
                for track_candidate_id in sorted(  # sort and reverse the iteration over the promoted track_candidates to not mess up the indexes
                    promoted_track_candidates,
                    reverse=True,
                ):
                    del self.track_candidates[track_candidate_id]

                # delete track candidate that were not found again
                # TODO: give track candidate multiple frame before being deleted
                self.track_candidates = [
                    track_candidate
                    for track_candidate in self.track_candidates
                    if track_candidate.is_detected()
                ]

        self.current_tracks_id = updated_tracks_ids.union(new_tracks_ids)
        self.current_track_candidates = current_track_candidates

        self.increment_age_and_delete_tracks(updated_tracks_ids)
        self.face_identify_tracks(img, self.current_tracks_id)

        # Try to identify tracks that got a new embedding
        self.identify_tracks(self.current_tracks_id)

        # Set the new_object_event if new tracks were found
        if len(new_tracks_ids) > 0:
            self.new_object_notifier.notify_new_object()

        self.count += 1

        # Write only the updated tracks on the db
        # TODO: only pass updated tracks so we don't serialize everytime
        if self.save_to_db:
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

    def get_matching_tracks(
        self, tracks: Dict[str, Track], detections: List[Detection], feature_vectors
    ):
        # Initialize cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))

        # Calculate cost for each pair of track and detection
        track_ids = list(tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = tracks[track_id]
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = self.encoder.compute_distance(
                    track.feature_vector, feature_vectors[j]
                )
                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    1 - self.lambda_value
                ) * (1 - iou_score)
        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices, cost_matrix

    def get_matching_track_candidates(self, detections: List, features_vectors):
        """
        Should pass the detections that are not matched with current tracks
        """
        # Initialize cost matrix
        cost_matrix = np.zeros((len(self.track_candidates), len(detections)))

        for i, track in enumerate(self.track_candidates):
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = self.encoder.compute_distance(
                    track.feature_vector, features_vectors[j]
                )
                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    1 - self.lambda_value
                ) * (1 - iou_score)
        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices, cost_matrix

    def add_track_candidate(self, detection, feature_vector: torch.Tensor):
        new_track_candidate = Track(
            track_id=detection.category,
            bbox=detection.bbox,
            feature_vector=feature_vector,
            distance=0,
            is_candidate=True,
        )
        new_track_candidate.set_is_detected()
        self.track_candidates.append(new_track_candidate)

    def promote_to_track(self, track_candidate_indice: int, feature_vector) -> str:
        """
        takes track indice and returns track_id
        """
        if len(self.track_candidates) <= track_candidate_indice:
            return IndexError(
                f"Can't find track candidate at indice {track_candidate_indice}"
            )

        track_candidate = deepcopy(self.track_candidates[track_candidate_indice])

        track_candidate.is_candidate = False
        track_id = self.generate_track_id(
            track_candidate._get_label()
        )  # for a track candidate, the label is the category
        track_candidate.change_track_id(track_id)
        self.tracks[track_id] = track_candidate

        # Add new track_id to the track_table
        self.track_ids_with_label[track_id] = [track_id]
        return track_id

    def clear_detected_track(self):
        for track in self.tracks.values():
            track.unset_is_detected()
        for track in self.track_candidates:
            track.unset_is_detected()

    def increment_age_and_delete_tracks(self, updated_tracks_ids=[]):
        # Remove or age out tracks that were not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks_ids:
                self.tracks[track_id].increment_age()

                # Optionally remove old tracks
                if self.tracks[track_id].age > self.max_age_track:
                    del self.tracks[track_id]
                    # TODO : also delete from track_ids_with_label ?

    def add_labeled_embedding(self, cmd: Dict):
        answer = {}
        for label, path in cmd.items():
            embeddings: List[np.ndarray] = []
            if not os.path.isdir(path):
                answer[label] = f"{path} is not a directory. can't add {label}"
            for file in os.listdir(path):
                if not (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    LOGGER.warning(
                        "Ignoring unsupported file type: %s. Only .jpg, .jpeg, and .png files are supported.",
                        file,
                    )  # TODO: integrate this warning in the answer of the do_cmd
                    continue
                im = Image.open(os.path.join(path, file)).convert("RGB")
                img = np.array(im)
                detections = self.detector.detect(img)

                if not detections:
                    LOGGER.warning(
                        "Can't find people on file: %s. Ignoring file.",
                        file,
                    )  # TODO: integrate this warning in the answer of the do_cmd
                    continue
                embeddings += self.encoder.compute_features(img, detections)
            self.labeled_person_embeddings[label] = embeddings
            answer[label] = (
                f"sucess adding {label}, found {len(embeddings)} embeddings."
            )
        return answer

    def delete_labeled_embedding(self, cmd):
        # TODO: check input here
        answer = {}
        for label in cmd:
            if label not in self.labeled_person_embeddings:
                answer[label] = f"can't find person {label}"
                continue
            del self.labeled_person_embeddings[label]
            answer[label] = f"success deleting: {label}"

        return answer

    def list_objects(self):
        answer = []
        for label, track_ids in self.track_ids_with_label.items():
            for track_id in track_ids:
                answer.append(self.generate_person_data(label=label, id=track_id))
        return answer

    def list_current(self):
        answer = {}
        for track_id in self.tracks:
            track = self.tracks[track_id]
            if track.is_detected():
                answer[track_id] = track.get_all()
        return answer

    def recompute_embeddings(self):
        self.face_identifier.compute_known_embeddings()
        self.compute_known_persons_embeddings()
        answer = "embeddings successfully recomputed"
        return answer

    @staticmethod
    def generate_person_data(label, id):
        return {"label": label, "id": id, "renamed": (id != label)}

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

    def compute_known_persons_embeddings(self):
        """
        Computes embeddings for known people from the picture directory.
        """

        if not self.path_to_known_persons:
            return
        if not os.path.exists(self.path_to_known_persons):
            LOGGER.warning(
                "Directory for known persons '%s' did not exist and was created.",
                self.path_to_known_persons,
            )
            os.makedirs(self.path_to_known_persons)

        all_entries = os.listdir(self.path_to_known_persons)
        directories = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(self.path_to_known_persons, entry))
        ]
        for directory in directories:
            label_path = os.path.join(self.path_to_known_persons, directory)
            embeddings = []
            for file in os.listdir(label_path):
                if not any(file.lower().endswith(suffix)
                           for suffix in (".jpg", ".jpeg", ".png")):
                    LOGGER.warning(
                        "Ignoring unsupported file type: %s. Only .jpg, .jpeg, and .png files are supported.",  # pylint: disable=line-too-long
                        file,
                    )
                    continue

                im = Image.open(label_path + "/" + file).convert("RGB")
                img_obj = ImageObject(viam_image=None, pil_image=im)

                detections = self.detector.detect(img_obj)

                if not detections:
                    LOGGER.warning(
                        f"Unable to find person in {directory}/{file}")
                    continue

                # Compute feature vectors for the current detections
                batched_features_vectors = self.encoder.compute_features(
                    img_obj, detections
                )
                list_of_tensors = list(batched_features_vectors.unbind(dim=0))
                embeddings += list_of_tensors
                LOGGER.debug(f"Added embedding for {directory}/{file}")

            if not embeddings:
                raise NoPersonDetectedError(
                    f"Unable to recognize any person in pictures from "
                    f"{directory}. Are you sure it contains a picture of a "
                    f"person?"
                )
            self.labeled_person_embeddings[directory] = embeddings

    def identify_tracks(self, track_ids):
        """
        Args:
        track_ids: track ids of tracks to be identified
        """
        if len(track_ids) < 1:
            return
        for track_id in track_ids:
            track = self.tracks[track_id]
            if track.has_label():
                continue
            track_embedding = track.get_embedding()
            found_match = False
            for label, embeddings in self.labeled_person_embeddings.items():
                if label == track.label_from_reid:
                    continue

                for embedding in embeddings:
                    feature_dist = self.encoder.compute_distance(
                        track_embedding, embedding
                    )

                    if feature_dist < self.reid_threshold:
                        found_match = True
                        track.label_from_reid = label
                        track.conf_from_reid = 1 - feature_dist

                        # old_label = track._get_label()
                        # track.relabel_reid_label(label)

                        # if only one track had the label 'old_label'
                        # if len(self.track_ids_with_label[old_label]) == 1:
                        #     if label not in self.track_ids_with_label:
                        #         self.track_ids_with_label[label] = (
                        #             self.track_ids_with_label.pop(old_label)
                        #         )
                        #     else:
                        #         self.track_ids_with_label[label].append(track_id)
                        #         del self.track_ids_with_label[old_label]

                        # else:
                        #     self.track_ids_with_label[old_label].remove(track_id)
                        #     if label not in self.track_ids_with_label:
                        #         self.track_ids_with_label[label] = [track_id]
                        #     else:
                        #         self.track_ids_with_label[label].append(track_id)

                        break
                if found_match:
                    break

    def face_identify_tracks(self, img: ImageObject, track_ids: List[str]):
        if len(track_ids) < 1:
            return
        for track_id in track_ids:
            track = self.tracks[track_id]
            new_face_id_label, new_face_id_conf = self.face_identifier.get_match(
                img, track
            )
            if new_face_id_label is not None:
                track.label_from_faceid = new_face_id_label
                track.conf_from_faceid = new_face_id_conf


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
