"""Lightweight SQLite-only DB prune script.

Deletes all rows from every table in `storage/pench_eye.db` except
for `camera_stations`. Uses only the Python standard library so it can be
run without the project's virtualenv.

Run from the repository root:
    python3 scripts/clear_sqlite_keep_cameras.py
"""
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "pench_eye.db"

if not DB_PATH.exists():
    print("Database not found at:", DB_PATH)
    sys.exit(1)

keep_tables = {"camera_stations"}

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = None
cur = conn.cursor()

try:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cur.fetchall()]
    print("Found tables:", tables)

    for t in tables:
        if t in keep_tables:
            print(f"Skipping table: {t}")
            continue
        print(f"Clearing table: {t}")
        cur.execute(f'DELETE FROM "{t}"')
        # Reset sqlite_sequence for AUTOINCREMENT tables
        try:
            cur.execute('DELETE FROM sqlite_sequence WHERE name=?', (t,))
        except sqlite3.OperationalError:
            pass

    conn.commit()
    try:
        cur.execute('PRAGMA wal_checkpoint(FULL);')
        cur.execute('VACUUM;')
    except Exception:
        pass

    print("Prune complete: camera_stations retained.")
finally:
    conn.close()
