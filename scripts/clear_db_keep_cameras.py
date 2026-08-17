"""Maintenance script: clear all rows from the application's database
except keep the `camera_stations` table intact.

Run from the repository root with the project's virtualenv active:

    python scripts/clear_db_keep_cameras.py

This script is destructive. It only operates on the configured DATABASE_URL_SYNC
from `app.core.config.settings` and will DELETE rows (not drop tables).
"""
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.core.database import Base

def main():
    db_url = settings.DATABASE_URL_SYNC
    print(f"Connecting to DB: {db_url}")

    engine = create_engine(db_url, future=True)

    # Ensure model metadata is imported so Base.metadata.tables is populated
    import app.models  # noqa: F401

    tables = list(Base.metadata.tables.keys())
    print("Found tables:", tables)

    # Tables to keep (do not delete rows from)
    keep_tables = {"camera_stations"}

    with engine.begin() as conn:
        try:
            for table in tables:
                if table in keep_tables:
                    print(f"Skipping table: {table}")
                    continue
                print(f"Clearing table: {table}")
                conn.execute(text(f'DELETE FROM "{table}"'))

            # For SQLite, optionally run WAL checkpoint/vacuum to shrink file
            if db_url.startswith("sqlite:"):
                try:
                    conn.execute(text("PRAGMA wal_checkpoint(FULL);"))
                    conn.execute(text("VACUUM;"))
                except Exception:
                    pass

            print("Database prune complete. Only camera_stations retained.")
        except SQLAlchemyError as exc:
            print("Error while pruning database:", exc)
            raise

if __name__ == "__main__":
    main()
