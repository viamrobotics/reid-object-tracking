import sqlite3
import time


def watch_database(db_path, table_name, interval=2):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    while True:
        # Fetch the data from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Clear screen or just print new results
        print("\033[H\033[J")  # Clear the terminal (on most systems)
        print(f"Data from {table_name}:")

        for row in rows:
            print(row)

        time.sleep(interval)  # Wait for the specified interval before refreshing


if __name__ == "__main__":
    watch_database(
        "/Users/robin@viam.com/object-tracking/re-id-object-tracking/tests/tracks.db",
        "tracks_ids_with_label",
    )
