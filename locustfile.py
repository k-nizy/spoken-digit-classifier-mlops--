"""Locust load test for the spoken digit prediction API.

The test talks HTTP to whatever sits in front of the API, so the same file
works against a local uvicorn and against the nginx load balancer that
docker-compose puts on port 8000.

Interactive web UI (open http://localhost:8089 to start the swarm):

    .venv/Scripts/python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000

Headless, 20 users for 60 s, prints the summary table used in the report:

    .venv/Scripts/python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 20 -r 5 -t 60s --csv loadtest --only-summary

The numbers to write down from the summary are:
    Requests    total requests sent
    Fails       should stay at 0 %
    Median (ms), 95%ile (ms), Max (ms)
    RPS         requests per second the stack sustained

The heavier run used for the scaling table (40 users, shorter think time):

    LOCUST_WAIT_MIN=0.5 LOCUST_WAIT_MAX=1.5 \
    .venv/Scripts/python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 40 -r 10 -t 60s --csv results/scale1_40u --only-summary

Environment variables
    LOCUST_UPLOAD=1         also send bulk /upload requests. Off by default,
                            because those write files and database rows on
                            every request and would dominate the run.
    LOCUST_MAX_CLIPS=200    how many test clips are loaded into memory.
    LOCUST_WAIT_MIN/MAX     think time in seconds between two requests of one
                            user (defaults 1 and 3). Lower values push more load
                            through with the same number of users. Both are
                            checked when the file is loaded: a value that is not
                            a number, a NaN/inf, a negative number, or MAX below
                            MIN stops the run with a message instead of quietly
                            changing what is being measured.
"""

import math
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "test"

# Read the sample clips once, at import time. Doing it inside the task would
# add disk I/O to the response time we are trying to measure.
MAX_CLIPS = int(os.environ.get("LOCUST_MAX_CLIPS", "200"))
CLIPS = []
for path in sorted(TEST_DIR.rglob("*.wav"))[:MAX_CLIPS]:
    label = int(path.stem.split("_")[0])
    CLIPS.append((path.name, path.read_bytes(), label))

if not CLIPS:
    raise SystemExit(
        f"no .wav files under {TEST_DIR} - run the data step of the notebook first"
    )

SEND_UPLOADS = os.environ.get("LOCUST_UPLOAD") == "1"


def think_time_bounds():
    """Read LOCUST_WAIT_MIN/MAX and refuse values that would silently change the
    experiment - a typo'd number, NaN/inf, a negative wait, or MAX below MIN.

    Locust itself would happily accept all of these: ``float("1o")`` raises deep
    inside the runner, ``between(nan, nan)`` never fires, and a swapped pair just
    makes every user wait the “wrong” amount of time. A load test whose think
    time is not what you think it is produces numbers that cannot be explained
    afterwards, so this fails at start-up instead.
    """
    raw = {
        "LOCUST_WAIT_MIN": os.environ.get("LOCUST_WAIT_MIN", "1"),
        "LOCUST_WAIT_MAX": os.environ.get("LOCUST_WAIT_MAX", "3"),
    }
    bounds = {}
    for name, text in raw.items():
        try:
            value = float(text)
        except ValueError:
            raise SystemExit(
                f"{name}={text!r} is not a number. Give a number of seconds, "
                f"for example 0.5 (the defaults are LOCUST_WAIT_MIN=1 "
                f"LOCUST_WAIT_MAX=3)."
            )
        if not math.isfinite(value):
            raise SystemExit(f"{name}={text!r} must be a finite number of seconds.")
        if value < 0:
            raise SystemExit(f"{name}={text!r} is negative; think time cannot be.")
        bounds[name] = value

    low = bounds["LOCUST_WAIT_MIN"]
    high = bounds["LOCUST_WAIT_MAX"]
    if high < low:
        raise SystemExit(
            f"LOCUST_WAIT_MAX={high} is smaller than LOCUST_WAIT_MIN={low}; "
            f"swap them. (MIN is the shortest pause between two requests, "
            f"MAX the longest.)"
        )
    return low, high


WAIT_MIN, WAIT_MAX = think_time_bounds()


class ClassifierUser(HttpUser):
    """One simulated client: predict mostly, check health now and then."""

    # Think time between two requests of the same user (default 1-3 s).
    wait_time = between(WAIT_MIN, WAIT_MAX)

    def on_start(self):
        # Fail fast and clearly if the API (or the load balancer) is not there.
        self.client.get("/health", name="/health (warmup)")

    @task(6)
    def predict(self):
        """The slow, real workload: one uploaded clip -> one digit."""
        filename, payload, _ = random.choice(CLIPS)
        self.client.post(
            "/predict",
            files={"file": (filename, payload, "audio/wav")},
            name="/predict",
        )

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def metrics(self):
        """Reads the model metrics and the database counts (the UI does this too)."""
        self.client.get("/metrics", name="/metrics")

    @task(1)
    def upload(self):
        """Bulk upload. Only runs with LOCUST_UPLOAD=1 (it writes to the database)."""
        if not SEND_UPLOADS:
            return
        filename, payload, label = random.choice(CLIPS)
        self.client.post(
            "/upload",
            files={"files": (filename, payload, "audio/wav")},
            data={"labels": str(label), "use_for_training": "false"},
            name="/upload",
        )
