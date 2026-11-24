"""SQLite storage for uploaded clips, predictions and retraining jobs.

Deliberately plain `sqlite3` from the standard library: one file, no server, easy
to inspect (`sqlite3 data/app.db ".tables"`). The API opens a short-lived
connection per operation, which is safe with FastAPI's thread pool.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_db_path = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    filename       TEXT NOT NULL,
    predicted_label INTEGER,
    confidence     REAL,
    source         TEXT,             -- 'single' or 'bulk'
    model_path     TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL,
    filename        TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    source_label    INTEGER,         -- label supplied by the user, NULL if unknown
    predicted_label INTEGER,
    confidence      REAL,
    used_for_training INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrain_jobs (
    id             TEXT PRIMARY KEY,
    status         TEXT NOT NULL,    -- running | finished | failed
    epochs         INTEGER,
    samples        INTEGER,
    accuracy_before REAL,
    accuracy_after  REAL,
    f1_after        REAL,
    model_path      TEXT,
    message         TEXT,
    started_at      TEXT,
    finished_at     TEXT
);
"""


def now():
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init(path):
    """Create the database file and tables if they do not exist yet."""
    global _db_path
    _db_path = Path(path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_db_path) as conn:
        # WAL lets readers work while another process writes. Needed when the
        # API runs as several containers sharing this file: without it the
        # load test produces "database is locked" errors.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    return _db_path


def connect():
    if _db_path is None:
        raise RuntimeError("db.init() must be called before using the database")
    conn = sqlite3.connect(_db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(cursor):
    return [dict(row) for row in cursor.fetchall()]


# --- predictions -------------------------------------------------------------
def add_prediction(filename, predicted_label, confidence, source, model_path):
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions (filename, predicted_label, confidence, source,"
            " model_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (filename, predicted_label, confidence, source, str(model_path), now()),
        )


def recent_predictions(limit=50):
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)))


def prediction_counts():
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT predicted_label, COUNT(*) AS n FROM predictions"
            " GROUP BY predicted_label ORDER BY predicted_label"))


# --- uploads -----------------------------------------------------------------
def add_upload(batch_id, filename, stored_path, source_label,
               predicted_label, confidence):
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO uploads (batch_id, filename, stored_path, source_label,"
            " predicted_label, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (batch_id, filename, str(stored_path), source_label,
             predicted_label, confidence, now()),
        )
        return cursor.lastrowid


def list_uploads(limit=100):
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,)))


def labelled_uploads():
    """Uploads that carry a true label - the only ones useful for training."""
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT * FROM uploads WHERE source_label IS NOT NULL"
            " ORDER BY id ASC"))


def mark_uploads_used(ids):
    if not ids:
        return
    with connect() as conn:
        conn.executemany("UPDATE uploads SET used_for_training = 1 WHERE id = ?",
                         [(i,) for i in ids])


def upload_counts():
    with connect() as conn:
        return {
            "total": conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0],
            "labelled": conn.execute(
                "SELECT COUNT(*) FROM uploads WHERE source_label IS NOT NULL"
            ).fetchone()[0],
            "used_for_training": conn.execute(
                "SELECT COUNT(*) FROM uploads WHERE used_for_training = 1"
            ).fetchone()[0],
        }


# --- retrain jobs ------------------------------------------------------------
def create_job(job_id, epochs):
    with connect() as conn:
        conn.execute(
            "INSERT INTO retrain_jobs (id, status, epochs, started_at)"
            " VALUES (?, 'running', ?, ?)", (job_id, epochs, now()))


def update_job(job_id, **fields):
    if not fields:
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    with connect() as conn:
        conn.execute(f"UPDATE retrain_jobs SET {columns} WHERE id = ?",
                     (*fields.values(), job_id))


def get_job(job_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM retrain_jobs WHERE id = ?",
                           (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit=10):
    with connect() as conn:
        return _rows(conn.execute(
            "SELECT * FROM retrain_jobs ORDER BY started_at DESC LIMIT ?", (limit,)))


def latest_job():
    jobs = list_jobs(1)
    return jobs[0] if jobs else None
