# AI usage log

This file records every step where AI assistance was used, the exact prompt,
what the AI produced, and how I changed it.

---

## Step 1 - Notebook: acquisition, preprocessing, model, evaluation

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> I am a student building my ALU Machine Learning Pipeline summative (45 pts).
> AI use is allowed only if I disclose it, so keep every step small and explain
> it so I can run and understand it. Do not write README prose or notebook
> explanation text; leave those as TODO for me.
> Problem: spoken digit recognition. Dataset: Free Spoken Digit Dataset (FSDD).
> Data type: sound (NOT tabular).
> Create the repo `spoken-digit-classifier-mlops` with README.md,
> notebook/<name>.ipynb, src/preprocessing.py, src/model.py, src/prediction.py,
> data/train, data/test, models/, app/API code, Dockerfile, locustfile.py,
> requirements.txt, docs/ai_usage_log.md.
> Step 1: notebook with data acquisition, preprocessing, model creation, model
> testing; one optimisation technique; at least 4 metrics; save the model to
> models/. Stop after each step so I can run it.
> After every step append to docs/ai_usage_log.md: the prompt, the output, and a
> line I fill in on how I changed it.

### What the AI produced

- `requirements.txt` - pinned-ish dependency list (numpy, pandas, matplotlib,
  librosa, soundfile, scikit-learn, tensorflow, jupyterlab).
- `.gitignore` - ignores `.venv/`, `data/raw/`, notebooks checkpoints and the
  API/UI runtime files added in later steps.
- `README.md` - heading skeleton only; every piece of prose is a `TODO`.
- `notebook/spoken_digit_classifier.ipynb` - 20 cells:
  1. setup / paths / seed
  2. data acquisition: download FSDD zip from GitHub, extract to `data/raw/`
  3. parse the file names (`<digit>_<speaker>_<index>.wav`), stratified 80/20
     split, copy the clips into `data/train/<digit>/` and `data/test/<digit>/`
  4. preprocessing: load at 8 kHz, pad or trim to 1 s, 64-band log-mel
     spectrogram, min-max normalise to 0..1
  5. two exploratory plots (spectrogram per digit, class balance)
  6. model creation: 3 small conv blocks with BatchNorm + dropout, then a
     dense head (10-class softmax)
  7. optimisation technique: **early stopping** (patience 8, best weights
     restored), plus a best-model checkpoint and a `ReduceLROnPlateau`
  8. evaluation: loss, accuracy, macro/weighted precision, recall and F1,
     classification report, confusion matrix and per-class accuracy
  9. saves `models/spoken_digit_cnn.h5`, `models/spoken_digit_cnn.keras` and
     `models/metrics.json`

### Issue found while verifying the notebook

The first run of the notebook scored 10% accuracy - every clip was predicted as
"6". Two causes: the model used `BatchNormalization`, which kept the network
from learning (a probe showed 55% train accuracy after 15 epochs, while the same
CNN built without batch norm reached 97% on the same 480 clips), and the 60/15
validation split was cut from class-ordered data, so the validation set held only
digits 8 and 9 and early stopping fired at epoch 9. The fixes: remove
`BatchNormalization`, keep dropout for regularization, and shuffle the training
set before splitting off validation. The rerun reached **0.988 test accuracy**
in 34 epochs.

### How I changed it

TODO: (me) write what you edited - e.g. number of epochs, LR, which plots you
kept, or a bug you fixed.

---

## Step 2 - Move the logic into src/

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> Step 1 works. Now do step 2: move the preprocessing, model building and
> prediction logic into src/preprocessing.py, src/model.py and
> src/prediction.py so the API can reuse them later, and append the entry to
> docs/ai_usage_log.md with a line for me to fill in.

### What the AI produced

- `src/__init__.py` - makes `src` an importable package.
- `src/preprocessing.py` - one pipeline, three levels of granularity:
  `load_audio()` (8 kHz, pad/trim to 1 s), `logmel_from_array()` (64 mel bands,
  dB, 0..1 scaling), `wav_to_logmel()` (file -> image). Plus `load_dataset()`
  which reads `data/<split>/<digit>/*.wav` into `(X, y, names)` for training,
  `prepare_batch()` for a list of uploaded files, and the shared constants
  (`INPUT_SHAPE = (64, 63, 1)`, `CLASS_NAMES`, `AUDIO_SUFFIXES`).
- `src/model.py` - `build_model()` (same CNN as the notebook),
  `train_model()` (early stopping + checkpoint + ReduceLROnPlateau, and an
  `initial_model_path` argument that warm-starts from an existing model - this
  is the retraining path), `save_model()`, `load_model()` and
  `evaluate_model()` which returns accuracy, loss, precision, recall, F1 (macro
  and weighted), per-class accuracy and the full classification report.
- `src/prediction.py` - `get_model()` caches the model in memory,
  `reset_model_cache()` drops it after retraining, `predict_spectrogram()` and
  `predict_file()` / `predict_files()` return `{digit, label, confidence,
  probabilities, filename}`.

### Verification run

Loaded `data/test` through `src/preprocessing.load_dataset` (600, 64, 63, 1),
scored the saved model with `src.model.evaluate_model` (accuracy 0.9883,
F1 0.9884, loss 0.0495 - identical to the notebook), predicted two clips through
`src.prediction.predict_file` (both correct, confidence 1.000), and ran a
2-epoch warm start from `models/spoken_digit_cnn.keras` to confirm the
retraining path saves a model.

### How I changed it

TODO: (me) write what you edited in src/ - e.g. renamed a function, changed a
constant, or added a check of your own.

---

## Step 3 - FastAPI service with a database and retraining

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> Step 2 works. Now do step 3: a FastAPI app that predicts one uploaded sound,
> accepts a bulk upload of multiple files with the results saved to a database,
> and exposes a retrain endpoint that warm-starts from my saved model using the
> uploaded data. Append the entry to docs/ai_usage_log.md with a line for me to
> fill in, and stop so I can run it.

### What the AI produced

- `app/db.py` - SQLite storage using only the standard library (`sqlite3`), so
  there is no extra dependency and the file can be inspected with any SQLite
  browser. Three tables: `predictions` (every prediction served),
  `uploads` (batch id, file path, given label, predicted label, confidence,
  whether it was used for training) and `retrain_jobs` (status, epochs, samples,
  accuracy before/after, message).
- `app/main.py` - the FastAPI application:
  - `GET /health` - uptime, model loaded, model path
  - `POST /predict` - one uploaded audio file -> digit, label, confidence and
    the full probability vector
  - `POST /upload` - many files at once; each file is stored on disk, predicted,
    and written to the database. Labels are optional (`labels=3,3,7` in the same
    order); without them the digit is taken from the file name, so unlabelled
    clips are still stored and scored but flagged as unusable for training
  - `POST /retrain` - warm-starts from `models/spoken_digit_cnn.keras` on the
    uploaded clips (`include_original=true` also replays `data/train`), evaluates
    before and after on the fixed test set, backs up the old weights, and runs in
    a background thread unless `background=false`
  - `GET /retrain/{job_id}` and `GET /retrain` - job status and history
  - `GET /uploads` and `GET /metrics` - the data the UI will draw in step 4
- `app/__init__.py`, and `fastapi`, `uvicorn`, `python-multipart` added to
  `requirements.txt`.

### Verification run (a throwaway database and a copy of the model)

- `GET /health` -> `{"status": "ok", "model_loaded": true, ...}`
- `POST /predict` with a real test clip -> digit 3, confidence 0.99999
- `POST /upload` twice (5 files with explicit labels, 5 files relying on the
  file-name label) -> 10 rows in `uploads`, all predictions correct
- `POST /retrain?epochs=25` -> job `finished`, 2410 clips (10 uploaded + 2400
  original), accuracy before 0.9883 -> after 0.9850, saved as a new model
- `POST /retrain` in the background + `GET /retrain/{job_id}` polling -> job
  went `running` -> `finished` without blocking the API
- `GET /metrics` -> database counts, predictions per class and the latest job

### Issue found and fixed while verifying

The first retrain run failed with `Cannot take the length of shape with unknown
rank`. The worker thread was reusing the model object cached at startup; loading
the model from disk inside the thread fixed it. Two more changes came out of
testing: fine-tuning uses a smaller learning rate (3e-4) instead of 1e-3, and a
retrained model is only promoted if it is not more than 0.5 accuracy points
worse than the serving one (otherwise the previous weights are restored and the
job says so) - retraining should never silently make production worse.

### How I changed it

TODO: (me) write what you edited in app/ - e.g. a different minimum number of
uploads before retraining, your own endpoint, or a change to the SQLite schema.

---

## Step 4 - Streamlit UI, and a fix for concurrent /predict requests

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> Step 3 works. Now do step 4: a Streamlit UI showing model up-time, three feature
> visualisations, and buttons to bulk-upload clips and trigger retraining, plus the
> log entry. Also fix the /predict temp-file collision so concurrent requests cannot
> overwrite each other's file.

### What the AI produced

- `app/ui.py` - the dashboard, in four tabs:
  - **Overview** - service status, human-readable uptime from `/health`, whether the
    model is loaded, upload count, the four trained metrics (accuracy, precision,
    recall, F1) from `/metrics`, the latest retraining job, and the database counters
  - **Feature insights** - three charts built from `data/test` with the same
    preprocessing module the model uses:
    1. the average log-mel spectrogram for each digit (2x5 grid), i.e. which
       frequencies each word puts energy into
    2. a boxplot of the mean spectral centroid per digit (how bright the sound is)
    3. energy (RMS) against clip length, coloured by digit
    Each has a TODO caption for the written interpretation
  - **Predict** - one clip -> digit, confidence and a probability bar chart
  - **Upload & retrain** - multi-file upload with an optional comma-separated label
    field and a use-for-training checkbox, a results table, then an epochs slider, an
    include-original-data checkbox and a retrain button that starts `POST /retrain`
    and polls the job until it finishes, showing accuracy before/after
- `requirements.txt` - added `streamlit` and `requests`.
- `app/main.py` - the `/predict` fix (see below).

### The bug: temp-file collision in `/predict`

`/predict` used to save every upload under its original name in a shared folder and
overwrite anything already there. Two requests with the same file name - which is
normal when several people send `recording.wav` - could therefore write to the same
path at the same time, and one request could decode the other request's audio.

The fix: `/predict` now creates a private file with `tempfile.mkstemp()` (a new name
the OS guarantees is unused) and deletes it in a `finally` block, so the response is
still labelled with the original name but every request has its own file. The bulk
upload path, which can also receive repeated names inside one batch, now opens files
with the exclusive `"xb"` mode and retries with a unique suffix on a clash.

### Verification run

- Concurrency: 12 simultaneous `POST /predict` calls, all sending a file called
  `7_george_0.wav` with the same audio -> all 12 returned digit 7 at confidence
  0.99999, and the temp folder was empty afterwards (no leftovers, no mix-ups).
- UI: the page was executed with Streamlit's own `AppTest` harness against a running
  API - no exceptions, all four tabs rendered, the metrics read 0.9883 accuracy and
  the three charts were built from `data/test`.
- Started the UI on `http://127.0.0.1:8501` against the API on port 8000; both
  answered HTTP 200.

### Issue found while verifying

Streamlit's test harness immediately caught a wrong dictionary key: the UI read
`metrics["uploads"]["uploads"]`, but `db.upload_counts()` returns `total`,
`labelled` and `used_for_training`. Fixed to `total` in both places. This is why the
page is executed once, rather than only started, before calling it done.

### How I changed it

TODO: (me) write what you edited in app/ui.py - e.g. your own chart, a different
layout, or the interpretation text for the three visualisations.
---

## Step 5 - Dockerfile and docker-compose

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> Step 4 works. Now do step 5: the Dockerfile and docker-compose setup to
> containerise the API and the UI. Docker is not installed yet, so write it, explain
> how to build and run it, and append the entry to docs/ai_usage_log.md with a line
> for me to fill in, then stop so I can install Docker and run it.

### What the AI produced

- `Dockerfile` - one image that runs either service, based on `python:3.12-slim`:
  - installs `libsndfile1` (decoding wav/flac/ogg for soundfile/librosa) and
    `libgomp1` (the OpenMP runtime TensorFlow links against); without these the
    container starts but every upload fails to decode
  - copies `requirements.txt` and installs it **before** copying the code, so
    editing a Python file does not re-download TensorFlow (the slow layer)
  - copies `src/`, `app/`, `models/`, `data/train/`, `data/test/` - the model to
    serve, the test set to evaluate a retrain, and the training set it replays
  - sets `MODEL_PATH` and `DATABASE_PATH` to paths inside the image
  - default command is uvicorn on port 8000; the UI container overrides it
- `.dockerignore` - keeps `.venv/`, `.git/`, `__pycache__/`, `*.db`,
  `data/uploads/`, `data/raw/`, `models/backup/`, the notebook and docs out of the
  build context. Without this the 1 GB `.venv` is uploaded to the daemon on every
  build.
- `docker-compose.yml` - two services from that one image:
  - `api`: publishes 8000, mounts the named volumes `uploads` (clips) and `dbdata`
    (SQLite), and has a healthcheck hitting `/health` every 10 s with a 60 s start
    period, because loading the model takes a few seconds
  - `ui`: publishes 8501, waits for `api` to be **healthy** before starting, and
    gets `API_URL=http://api:8000` - the compose service name, since inside a
    container `localhost` is the UI container itself
  - both healthchecks use the standard-library `urllib`, because the slim image has
    no `curl` or `wget`

  *(step 6 changed this: the `api` no longer publishes a host port - nginx does -
  and the UI talks to nginx instead of directly to `api`.)*

Why the volumes matter: uploaded clips are stored on disk *and* referenced by the
SQLite rows, so retraining needs both to still exist after a restart. The retrained
model is written inside the container (`/app/models`), so each `docker compose up`
starts from the model trained in the notebook - that is deliberate, it keeps the
container reproducible.

### Verification run

Docker is not installed on this machine yet, so the image has not been built. It
was emulated instead: exactly the files the Dockerfile copies (`src`, `app`,
`models`, `data/train`, `data/test`, `requirements.txt`) were placed in a clean
folder and the container commands were run against it with the same `MODEL_PATH`,
`DATABASE_PATH` and `API_URL` values docker-compose sets.

What was checked offline:

- `docker-compose.yml` parses as YAML and has both services, both volumes, the UI's
  `depends_on: service_healthy` condition and its `API_URL`
- every path the Dockerfile copies exists (`src`, `app`, `models`, `data/train`,
  `data/test`, `requirements.txt`)
- `db.init()` creates the database's parent directory, so `DATABASE_PATH=/app/data/db/app.db`
  works even when the volume is empty
- the model-loading and path overrides were already exercised in step 3 with
  `MODEL_PATH` / `DATABASE_PATH` on a different port

Also checked against the emulated image, over HTTP:

| Step | Result |
| --- | --- |
| `GET /health` | 200, model loaded from the copied `models/` |
| `POST /predict` | digit 4, confidence 0.9998 |
| `POST /upload` (5 labelled clips) | 5 saved, 5 predicted correctly, files on disk, rows in `data/db/app.db` |
| `POST /retrain?epochs=2&include_original=true&background=false` | job `finished` in 32 s on 2405 clips (5 uploaded + 2400 replayed from `data/train`), model rewritten, backup saved |
| `GET /metrics` | current model, upload counts, latest job |
| UI from the image folder | rendered with no exceptions, all three charts and both retrain figures |

### Issue found while testing

Retraining once exposed a UI bug that the step-4 check could not see: the Overview
tab read `latest["job_id"]`, but the retrain job row is keyed `id`, so the dashboard
crashed with `KeyError: job_id` as soon as a retrain existed. The step-4 test used an
empty database, where that section is skipped entirely. Fixed to `latest["id"]` and
the page then rendered the job summary correctly.

### How I changed it

TODO: (me) write what you edited in the Dockerfile or docker-compose.yml - e.g. a
different base image, a volume for the retrained model, or a pinned port.

---

## Step 6 - Locust flood test and an nginx load balancer

**Date:** 2026-11-24
**AI tool:** Freebuff coding agent

### Prompt used

> Step 6: the flood simulation. Write `locustfile.py` for the API, put a load
> balancer in front of it so I can run `docker compose up --scale api=1|2|3`,
> and give me the table I have to fill in with latency at 1, 2 and 3
> containers. Docker is still not installed here, so make the code correct and
> verified as far as possible without it, tell me exactly what is left to run,
> and append the entry to docs/ai_usage_log.md with a line for me to fill in.

### What the AI produced

- `locustfile.py` - one `HttpUser` called `ClassifierUser`:
  - the sample clips under `data/test/` are read **once at import time**, so
    disk I/O is not counted in the response time the test reports
  - task weights: `/predict` 6, `/health` 2, `/metrics` 1, `/upload` 1
  - `/upload` does nothing unless `LOCUST_UPLOAD=1`, because every call writes a
    file and two database rows and would swamp the measurement
  - `wait_time = between(1, 3)` (think time), and an `on_start` warm-up request
    so a missing API fails immediately instead of after 60 s of retries
  - `LOCUST_MAX_CLIPS` caps how many clips are kept in memory
- `nginx.conf` - a load balancer on port 80:
  - `upstream spoken_digit_api { zone ...; resolver 127.0.0.11 valid=10s;
    server api:8000 resolve; keepalive 32; }` - the `resolve` keyword makes
    nginx ask Docker's DNS again every 10 s, so adding a replica with
    `--scale api=3` is picked up without restarting nginx. It needs nginx >=
    1.27.3, hence the pinned `nginx:1.29-alpine` image.
  - `client_max_body_size 50m` (bulk uploads), `proxy_read_timeout 120s` (a cold
    model load is slow), and `location = /lb-health` used by the container
    healthcheck so nginx can be unhealthy independently of the API.
- `docker-compose.yml` - now three services from two images: the `api` no longer
  publishes a host port (it only `expose`s 8000, otherwise `--scale` fails with
  "port is already allocated"), a new `nginx` service publishes `8000:80` and
  waits for every `api` replica to be healthy, and the UI's `API_URL` now points
  at `http://nginx:80` so the dashboard also goes through the balancer.
- `app/db.py` - `PRAGMA journal_mode=WAL` when the database is created. With
  three API containers writing one SQLite file, the default rollback journal
  blocks readers; WAL lets them read while the writer works. The file-level
  `timeout=10` in `connect()` was already acting as `busy_timeout`.
- `requirements.txt` - added `locust>=2.29`, so `pip install -r requirements.txt`
  is still the only install command.

### Verification run

What could be checked without Docker:

| Check | Result |
|---|---|
| `locust -f locustfile.py --list` | `ClassifierUser` discovered, no import errors |
| Real run: 10 users, spawn rate 5/s, 20 s, against a local uvicorn on :8000 | 40 requests, **0 failures (0.00 %)** |
| POST /predict under that load | median 710 ms, 95%ile 11 s, max 11.4 s, 2.29 req/s |
| GET /metrics under that load | avg 205 ms, 4 requests, 0 failures |
| database after the run | `app.db-wal` written (136 KB) - WAL mode is active and the API kept serving |
| fresh database + `POST /upload` with two labelled clips, via `requests` (the way the UI sends them) | 200, both classified correctly, `/metrics` reports `total 2, labelled 2`, two prediction rows, 0 server errors |
| `docker-compose.yml` parsed as YAML | services `api`, `nginx`, `ui`; api `expose: 8000` and no `ports`; nginx `8000:80`; ui `8501:8501` depending on `nginx: service_healthy` |
| file hygiene | `docker-compose.yml`, `nginx.conf`, `locustfile.py`, `Dockerfile`: no CRLF, no tabs |

The high /predict tail is expected and is worth explaining in the README: ten
simultaneous clients, one uvicorn process and TensorFlow using every core, so
requests queue behind each other. That is exactly what the 1/2/3-container
comparison is meant to show.

### Not verified yet

- `nginx -t` never ran, because nginx is not installed on this machine; the
  config is only reasoned about.
- `docker compose up --build` and `--scale api=1|2|3` never ran, because Docker
  is not installed. The latency table therefore still has empty cells.

### How I changed it

TODO: (me) write what you edited in locustfile.py, nginx.conf or docker-compose.yml
- e.g. a different number of users or a longer run time, a different wait time,
or a second nginx worker process.

---

## Step 7 - Install Docker and prove the step-6 stack at 3 replicas

**Date:** 2026-11-25
**AI tool:** Freebuff coding agent

### Prompt used

> Before I install Docker, review the new step 6 files (locustfile.py,
> nginx.conf, docker-compose.yml and the WAL change in app/db.py) with fresh
> eyes and make sure nothing will break my first `docker compose up --scale
> api=3`. install the docker

### What the AI produced

No code changes were needed: the review found nothing that would break the
first scaled start. What it did instead:

- installed **Docker Desktop 29.9.3** with Compose **v5.7.1**. `winget` hung on
  the download, so the official installer was downloaded directly and run with
  `install --quiet --accept-license --backend=wsl-2`.
- checked the one thing in `nginx.conf` that is easy to get wrong:
  `resolver ...` **is** allowed inside an `upstream` block since nginx 1.27.3
  (it is what `server api:8000 resolve` needs), and 1.29 is what is pinned.
- confirmed the file line endings are LF (`file` plus a byte count; an earlier
  `grep -c $'\r'` was matching the letter "r" and was misleading).
- confirmed the container resolves the same package versions as the working
  local `.venv` (tensorflow 2.21.0, numpy 2.5.5, pandas 3.1.1).

### Verification run

| Check | Result |
|---|---|
| `docker compose config --quiet` | valid, services `api`, `nginx`, `ui` |
| `docker compose --dry-run up --scale api=3 -d` | plans api-1/2/3, waits for **all three** healthy, then nginx, then ui; no host-port clash |
| `nginx -t` on `nginx:1.29-alpine` | syntax OK |
| nginx started with no api container at all | container stays running, `/lb-health` 200, upstream 502 - it does not block on DNS |
| three containers, one SQLite file, 200 inserts each | 600 rows, `journal_mode = wal`, no "database is locked" |
| round robin through nginx with a stub replica (12 requests) | exactly 3 distinct replicas, evenly spread |
| scale 3 -> 1 without restarting nginx | 10/10 HTTP 200, 1 distinct replica (no 502 storm) |
| `docker compose build` | image `spoken-digit-api:local`, 1.07 GB |
| `docker compose up -d --scale api=3` | 5 containers healthy in ~50 s |
| `/`, `/health`, `/metrics`, `/uploads` through nginx | all 200 |
| `/predict` through nginx (`data/test/4/4_george_1.wav`) | digit 4, confidence 0.9998 |
| `/upload` 5 labelled clips through nginx | 5/5 correct, `/uploads` counts `total 5, labelled 5` |
| 9 `/health` requests, counted per container | api-1: 10, api-2: 10, api-3: 9 |
| `/retrain?epochs=1` in a container | `finished` in 39 s, 2405 clips, 0.9883 -> 0.9833, backup written |
| api logs | only the known `cuInit: UNKNOWN ERROR (303)` no-GPU warning; no 500s, no tracebacks |

The stack was then stopped with `docker compose down -v`, so the database and
the uploaded clips used for these tests are gone and the demo starts clean.

### Not verified yet

- The 1/2/3-container Locust table is still empty: the load test needs to be run
  three times by hand against the running stack (`docs/WHATS_LEFT.md` 2.2).
- Render deployment and the demo video, which need an account and a camera.

### How I changed it

TODO: (me) write what you changed after reading this - e.g. a different nginx
image tag, a higher `client_max_body_size`, or a `--scale` value other than 3.

## Step 8 - Heavier flood test: 40 users, shorter think time

**Date:** 2026-11-26
**AI tool:** Freebuff coding agent

### Prompt used

> My flood-test table shows 3 containers no better than 2, which will get
> questioned on camera. Re-run it with more load (about 40 users, shorter think
> time), keep the old CSVs, and update the README table and interpretation so
> the scaling story is honest and clear.

### What the AI produced

- `locustfile.py`: think time is now read from `LOCUST_WAIT_MIN` /
  `LOCUST_WAIT_MAX` (defaults still 1 s and 3 s), so the same file can run a
  light and a heavy test without editing code.
- Three new runs at 40 users, spawn rate 10/s, 60 s each, think time 0.5-1.5 s,
  written to `results/scale1_40u`, `scale2_40u`, `scale3_40u` and
  `results/scale3b_40u`. The old 20-user CSVs were left untouched.
- Each replica was warmed up with 30-60 `/predict` calls before the clock
  started. The first attempt without that produced one 502 in the 2-container
  run (a just-started replica behind nginx) and a `max` of ~6 s that was a cold
  TensorFlow session, not a scaling effect.
- README: new numbers, the warm-up step in the commands, and a written
  interpretation ("Reading the table").

### Verification run

| Check | Result |
|---|---|
| `results/scale1_40u_stats.csv` | 1425 requests, 0 fails, median 460 ms, 95% 1200 ms, 24.4 req/s |
| `results/scale2_40u_stats.csv` | 1600 requests, 0 fails, median 210 ms, 95% 1200 ms, 27.3 req/s |
| `results/scale3_40u_stats.csv` | 1428 requests, 0 fails, median 330 ms, 95% 1700 ms, 24.7 req/s |
| `results/scale3b_40u_stats.csv` (repeat) | 1438 requests, 0 fails, median 340 ms, 95% 1500 ms, 25.0 req/s |
| 3 replicas, two runs | agree with each other, so the plateau at 3 is reproducible |
| host core count (`nproc`) | 8 logical CPUs; each container runs TensorFlow with 2 intraop threads |
| idle `docker stats` after the runs | one api container ~434-510 MiB, so 3 replicas fit in RAM on this laptop |

That first set was the wrong experiment, which the numbers showed: 40 users with
0.5-1.5 s think time only offer ~25 req/s, so 1, 2 and 3 containers all landed at
24-27 req/s. CPU never came close to the 8-core ceiling (267 % of 800 %), i.e. the
load generator was measuring itself. So the test was repeated with 0.1-0.3 s think
time, where the server is the bottleneck:

| Run | Result |
|---|---|
| `scale1_40u_fast_stats.csv` | 1287 requests, 0 fails, median 1700 ms, 95% 2700 ms, 22.8 req/s |
| `scale2_40u_fast_stats.csv` | 1927 requests, 0 fails, median 780 ms, 95% 2500 ms, 32.4 req/s |
| `scale2b_40u_fast_stats.csv` | 1384 requests, 0 fails, median 1300 ms, 95% 3200 ms, 23.7 req/s |
| `scale3_40u_fast_stats.csv` | 1960 requests, 0 fails, median 760 ms, 95% 2700 ms, 32.9 req/s |
| `scale3b_40u_fast_stats.csv` | 1575 requests, 1 fail, median 920 ms, 95% 3400 ms, 26.9 req/s |
| `scale3_cpu_40u_fast_stats.csv` | 1341 requests, **5 fails**, median 970 ms, 95% 4100 ms, 22.8 req/s |
| CPU sampled during the last run (`results/cpu_3_run.txt`) | each api container ~230 %, together ~690 % of the 800 % available |
| `docker compose logs api` for the 5 failures | `sqlite3.OperationalError: database is locked` -> HTTP 500 on `/predict` |
| `results/scale2a_40u_fast_*` | excluded: 796 x 502 in a 10 s window, Docker recreated nginx/api mid-run, no api container was actually down |

Conclusion for the report: two replicas are a real win (median 1700 -> 780 ms,
throughput 22.8 -> 32.4 req/s, repeat run 23.7 as the error bar); a third replica
is at best a tie and usually worse, because the 8-core host is saturated and the
shared SQLite file starts returning "database is locked".

### Not verified yet

- Whether a third replica helps on a machine with more cores (or a GPU). Only
  this 8-core laptop was measured.
- Whether moving the database to PostgreSQL removes the 500s at 3 replicas - that
  is a suggestion, not something that was implemented or tested.
- A distributed Locust run (`--master`/`--worker`), so the load generator is not
  part of the measurement.

### How I changed it

TODO: (me) reword the "Reading the table" paragraph in your own voice, and say
whether you would test a 4th container, a GPU host or PostgreSQL next.

## Step 9 - Confirm retraining at one API container, and harden the think-time setting

**Date:** 2026-11-26
**AI tool:** Freebuff coding agent

### Prompt used

> Scale the stack back to one API container and run a one-epoch retrain to
> confirm it still works at that size, since that is what I will record. Also
> give the locustfile think-time change a last look so a bad value fails clearly
> instead of silently.

### What the AI produced

- `docker compose up -d --scale api=1`, then `POST /retrain?epochs=1`.
- `locustfile.py`: the two `LOCUST_WAIT_*` values are now read by
  `think_time_bounds()` when the file is loaded, which stops the run with a clear
  message for a value that is not a number, a NaN/inf, a negative number, or a
  maximum below the minimum. Before this, `LOCUST_WAIT_MIN=1o` crashed deep inside
  Locust, and `LOCUST_WAIT_MIN=5 LOCUST_WAIT_MAX=1` quietly reversed the two.

### Verification run

| Check | Result |
|---|---|
| `docker compose up -d --scale api=1` | 3 containers healthy (api, nginx, ui) |
| `POST /retrain?epochs=1` with 0 uploads | HTTP 400, "need at least 5 labelled uploads" - the guard works |
| `POST /upload` with 5 labelled clips | 5/5 saved, 5/5 predicted correctly, `labelled: 6` total |
| `POST /retrain?epochs=1` at 1 container | `finished` in 74 s (10:41:02 -> 10:42:16), 2406 clips (6 uploaded + 2400 original), accuracy 0.9883 -> 0.9883, F1 0.9884, backup `spoken_digit_20261126-104134.keras` |
| `/uploads` after the retrain | `used_for_training: 6` |
| `/predict` after the retrain | 7 at 0.99998 confidence, model reloaded |
| `POST /retrain?epochs=5` at 1 container (the setting in the video script) | `finished` in 67 s (10:44:37 -> 10:45:44), 2406 clips, accuracy 0.9883 -> 0.9867, F1 0.9867, backup written |
| bad `LOCUST_WAIT_MIN=1o` | exit 1, "is not a number. Give a number of seconds..." |
| swapped `LOCUST_WAIT_MIN=5 LOCUST_WAIT_MAX=1` | exit 1, "is smaller than ... swap them" |
| `LOCUST_WAIT_MIN=-1` | exit 1, "is negative; think time cannot be" |
| `LOCUST_WAIT_MIN=nan`, `LOCUST_WAIT_MAX=inf` | exit 1, "must be a finite number of seconds" |
| valid `0.5/1.5`, `2/2`, and unset | accepted, bounds 0.5-1.5, 2.0-2.0, 1.0-3.0 |
| `python -m locust` with a bad value | exit 1 with the same message, before any request is sent |

### Not verified yet

- The full demo path through the Streamlit UI - only the API endpoints were
  exercised here.
- The check is indicative only: the six clips uploaded for it came from
  `data/test`, which is also part of the 600-clip accuracy set. In the real demo
  the uploads are your own recordings, so the "after" number will differ.

### How I changed it

TODO: (me) note anything you changed in the demo run, e.g. the number of epochs
you picked so the retrain fits the video.
