from src.tracker.track import Track
from typing import Dict
import sqlite3
import numpy as np
from src.config.config import TracksManagerConfig

# TODO: make it more maintanable and have map[name_of_table, list[columns]]
tables: Dict[str, str] = {
    "tracks": """
        CREATE TABLE IF NOT EXISTS tracks(
            id CHAR PRIMARY KEY,
            bbox BLOB,
            embedding BLOB
        )
    """,
    "category_count": """
        CREATE TABLE IF NOT EXISTS category_count(
            id CHAR PRIMARY KEY,
            count INTEGER
        )
    """,
}


class TracksManager:
    # TODO: keep track of the updated tracks since last sync so we don't serialize and write everything
    def __init__(self, config: TracksManagerConfig):
        self.tracks_on_disk: Dict[str, Track] = {}
        self.category_count_on_disk: Dict[str, int] = {}
        self.path_to_database = config.path_to_db.value
        self.save_period = config.save_period
        self.con: sqlite3.Connection = sqlite3.connect(self.path_to_database)
        instantiate_table("tracks", self.con)
        instantiate_table("category_count", self.con)

    def close(self):
        self.con.close()

    def parse_tracks_on_disk(self):
        # Fetch all track records from the database, including bbox
        res = self.con.execute("SELECT * FROM tracks").fetchall()

        for serialized_track in res:
            track = self.get_new_track_from_bytes(serialized_track)
            self.tracks_on_disk[track.track_id] = track

    @staticmethod
    def get_new_track_from_bytes(serialized_track: tuple):
        assert len(serialized_track) == 3
        track_id, bbox_blob, embedding_blob = serialized_track
        return Track(
            track_id=track_id,
            bbox=np.frombuffer(bbox_blob, dtype=int).reshape(
                4,
            ),
            feature_vector=np.frombuffer(embedding_blob, dtype=float).reshape(1, 128),
            distance=None,
        )

    def get_tracks_on_disk(self):
        return self.tracks_on_disk

    def write_tracks_on_db(self, tracks: Dict[str, Track]):
        for track_id, track in tracks.items():
            # Get the serialized data from each track
            track_id, bbox_blob, embedding_blob = track.serialize()

            # Insert the track data into the database
            self.con.execute(
                "INSERT OR REPLACE INTO tracks(id, bbox, embedding) VALUES(?, ?, ?)",
                (
                    track_id,
                    bbox_blob,
                    embedding_blob,
                ),
            )

        self.con.commit()

    def parse_category_count_on_disk(self):
        res = self.con.execute("SELECT * FROM category_count").fetchall()
        for category, count in res:
            self.category_count_on_disk[category] = count

    def get_category_count_on_disk(self):
        return self.category_count_on_disk

    def write_category_count_on_db(self, category_count: Dict[str, int]):
        for category, count in category_count.items():
            self.con.execute(
                "INSERT OR REPLACE INTO category_count(id, count) VALUES(?, ?)",
                (category, count),
            )
        self.con.commit()


def instantiate_table(table_name: str, con: sqlite3.Connection):
    con.execute(tables.get(table_name))
