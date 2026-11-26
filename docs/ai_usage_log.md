# AI usage log

Every step where AI was used: the prompt, what it produced, and how I changed it.
Nine steps, one short entry each, in the order they were done.

**Summary:** AI wrote a first draft of the code; I ran every step, reviewed every
file, wrote all the prose, and changed: TODO: (me) list the changes you made,
e.g. the settings you re-ran, the bugs you fixed, the text you rewrote.

## Step 1 - Notebook: data, preprocessing, model, evaluation  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "I am a student building my ALU Machine Learning Pipeline summative
(45 pts). AI use is allowed only if I disclose it ... Do not write README prose or
notebook explanation text; leave those as TODO for me ... Step 1: notebook with
data acquisition, preprocessing, model creation, model testing; one optimisation
technique; at least 4 metrics; save the model to models/."

**What the AI produced:** `requirements.txt`, `.gitignore`, a README skeleton
whose every paragraph is a `TODO`, and `notebook/spoken_digit_classifier.ipynb`
(20 cells): download FSDD and split it 80/20 stratified by digit, load at 8 kHz,
pad or trim to 1 s, build a 64-band log-mel spectrogram scaled to 0..1, train a
CNN of three conv blocks plus dropout into a 10-way dense head, use early stopping
(patience 8, best weights restored) with a best-model checkpoint and
`ReduceLROnPlateau`, then report loss, accuracy, macro and weighted precision,
recall and F1, a classification report and a confusion matrix, and save `.h5`,
`.keras` and `metrics.json`.

**Problem found:** the first run scored 10% - every clip predicted "6". Batch
normalisation stopped the network learning, and the validation split was cut from
class-ordered data, so it held only digits 8 and 9 and early stopping fired at
epoch 9. Removing batch norm (keeping dropout) and shuffling before the split
reached **0.9883** accuracy in 34 epochs.

**How I changed it:** TODO: (me) what I edited - epochs, learning rate, which
plots I kept, or a bug I fixed.

## Step 2 - Move the logic into src/  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "Step 1 works. Now do step 2: move the preprocessing, model building
and prediction logic into src/preprocessing.py, src/model.py and
src/prediction.py so the API can reuse them later ..."

**What the AI produced:** `src/preprocessing.py` (one pipeline: `load_audio`,
`logmel_from_array`, `wav_to_logmel`, plus `load_dataset` and `prepare_batch` and
the shared constants `INPUT_SHAPE`, `CLASS_NAMES`), `src/model.py`
(`build_model`, `train_model` with early stopping and an `initial_model_path` for
warm starts, `save_model`, `load_model`, `evaluate_model`) and
`src/prediction.py` (`get_model`, `reset_model_cache`, `predict_file`,
`predict_files`). Checked against the notebook: the same 600 test clips gave
accuracy 0.9883, F1 0.9884, loss 0.0495, and a 2-epoch warm start saved a model.

**How I changed it:** TODO: (me) what I edited in src/ - a renamed function, a
changed constant, a check I added.

## Step 3 - FastAPI service with a database and retraining  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "Step 3: a FastAPI app that predicts one uploaded sound, accepts a
bulk upload of multiple files with the results saved to a database, and exposes a
retrain endpoint that warm-starts from my saved model using the uploaded data."

**What the AI produced:** `app/db.py`, SQLite through the standard library with
three tables (`predictions`, `uploads`, `retrain_jobs`), and `app/main.py` with
`/health`, `/predict`, `/upload` (labels optional, otherwise taken from the file
name), `/retrain` (warm start, replays `data/train`, scores before and after on the
test set, backs up the old weights), `/retrain/{id}`, `/uploads` and `/metrics`.
Checked on a throwaway database: predict returned digit 3 at 0.99999, two uploads
stored 10 correct rows, a 25-epoch retrain finished on 2410 clips (0.9883 ->
0.9850), and a background job went running -> finished without blocking.

**Problems found:** the first retrain failed with "Cannot take the length of shape
with unknown rank" because the worker thread reused the model cached at startup -
loading the model inside the thread fixed it. Fine-tuning now uses 3e-4 instead of
1e-3, and a retrained model is promoted only if it is at most 0.5 accuracy points
worse than the serving one, so retraining cannot silently make production worse.

**How I changed it:** TODO: (me) what I edited in app/ - the minimum number of
uploads, an endpoint of my own, a change to the schema.

## Step 4 - Streamlit UI, and a fix for concurrent /predict requests  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "Step 4: a Streamlit UI showing model up-time, three feature
visualisations, and buttons to bulk-upload clips and trigger retraining ... Also
fix the /predict temp-file collision so concurrent requests cannot overwrite each
other's file."

**What the AI produced:** `app/ui.py` with four tabs - Overview (status, uptime,
the four metrics, latest job, database counters), Feature insights (mean log-mel
spectrogram per digit, spectral-centroid boxplot, energy against clip length, each
with a `TODO` caption), Predict, and Upload & retrain (batch upload, epoch slider,
polling of the job). The `/predict` bug: every upload was saved under its original
name in one shared folder, so two people sending `recording.wav` could overwrite
each other; it now uses `tempfile.mkstemp()` and deletes the file in a `finally`
block, and the bulk path opens files exclusively (`"xb"`) with a unique suffix on
a clash. 12 simultaneous requests with the same file name all returned digit 7.

**Problem found:** Streamlit's `AppTest` harness caught the UI reading
`metrics["uploads"]["uploads"]` where the database returns `total`; the page
rendered without errors after the fix.

**How I changed it:** TODO: (me) what I edited in app/ui.py - my own chart, a
different layout, the interpretation text for the three charts.

## Step 5 - Dockerfile and docker-compose  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "Step 5: the Dockerfile and docker-compose setup to containerise the
API and the UI. Docker is not installed yet, so write it, explain how to build and
run it ..."

**What the AI produced:** a `Dockerfile` on `python:3.12-slim` that installs
`libsndfile1` and `libgomp1` (without them every upload fails to decode or
TensorFlow cannot load), installs `requirements.txt` before copying code so
editing a file does not re-download TensorFlow, copies `src`, `app`, `models`,
`data/train`, `data/test`, and defaults to uvicorn on 8000. Plus a `.dockerignore`
(keeps the 1 GB `.venv` out of the build context) and a `docker-compose.yml` with
an `api` service on volumes for uploads and SQLite, a `ui` service that waits for
it to be healthy and gets `API_URL`, both health checks using `urllib` because the
slim image has no curl.

**Verified without Docker** (it was not installed yet): the compose file parsed,
every copied path existed, and the container commands were run against a clean
folder - health, predict, a 5-clip upload, a 2-epoch retrain on 2405 clips in 32 s
and the UI all worked.

**Problem found:** retraining exposed a crash the step-4 check could not see - the
Overview tab read `latest["job_id"]` but the row is keyed `id`, so the dashboard
died with `KeyError` as soon as a retrain existed (`KeyError` never fired on the
empty database used before). Fixed to `latest["id"]`.

**How I changed it:** TODO: (me) what I edited in the Dockerfile or
docker-compose.yml - base image, a volume, a pinned port.

## Step 6 - Locust flood test and an nginx load balancer  (2026-11-24)

**Tool:** Freebuff coding agent.

**Prompt:** "Step 6: the flood simulation. Write locustfile.py for the API, put a
load balancer in front of it so I can run docker compose up --scale api=1|2|3, and
give me the table I have to fill in with latency at 1, 2 and 3 containers."

**What the AI produced:** `locustfile.py` (sample clips read once at import so
disk I/O is not timed; weights `/predict` 6, `/health` 2, `/metrics` 1, `/upload`
1 but gated behind `LOCUST_UPLOAD=1` because it writes files and rows; think time
1-3 s; a warm-up request on start). `nginx.conf` as a balancer with
`server api:8000 resolve` and a Docker DNS resolver so `--scale api=3` is picked
up without restarting nginx (needs nginx >= 1.27.3, hence `1.29-alpine`), a 50 MB
body limit and a `/lb-health` endpoint. `docker-compose.yml` became three services
(`api` only `expose`s 8000, otherwise `--scale` fails with "port is already
allocated"), and `app/db.py` now sets `PRAGMA journal_mode=WAL` so three writers do
not block readers.

**Verified without Docker:** `locust --list` found the user class, a 10-user run
made 40 requests with 0 failures, the WAL side file appeared, and the compose file
parsed. Not verified: `nginx -t` and the 1/2/3-container table, which needed a
running Docker.

**How I changed it:** TODO: (me) what I edited in locustfile.py, nginx.conf or
docker-compose.yml - number of users, run length, wait time, worker processes.

## Step 7 - Install Docker and prove the step-6 stack at 3 replicas  (2026-11-25)

**Tool:** Freebuff coding agent.

**Prompt:** "Before I install Docker, review the new step 6 files ... with fresh
eyes and make sure nothing will break my first docker compose up --scale api=3.
install the docker"

**What the AI produced:** no code changes - the review found nothing that would
break the first scaled start. It installed Docker Desktop 29.9.3 with Compose
v5.7.1 (the direct installer, because `winget` hung), confirmed `resolver` is
allowed inside an `upstream` block since nginx 1.27.3, confirmed LF line endings,
and confirmed the container resolves the same package versions as the local
`.venv`.

**Verified:** `nginx -t` passed; `--dry-run up --scale api=3` planned api-1/2/3
then nginx then ui with no port clash; three containers healthy in about 50 s;
`/predict` gave digit 4 at 0.9998; 5 uploaded clips all correct; 9 requests split
10/10/9 across the replicas; scaling 3 -> 1 gave 10/10 replies with no nginx
restart; a 1-epoch retrain in a container finished in 39 s on 2405 clips
(0.9883 -> 0.9833). The stack was then cleared with `docker compose down -v`.

**How I changed it:** TODO: (me) what I changed after reading this - an nginx tag,
a bigger body limit, a `--scale` value other than 3.

## Step 8 - Heavier flood test: 40 users, shorter think time  (2026-11-26)

**Tool:** Freebuff coding agent.

**Prompt:** "My flood-test table shows 3 containers no better than 2, which will
get questioned on camera. Re-run it with more load (about 40 users, shorter think
time), keep the old CSVs, and update the README table and interpretation so the
scaling story is honest and clear."

**What the AI produced:** `LOCUST_WAIT_MIN` / `LOCUST_WAIT_MAX` so one file can
run a light and a heavy test, new 40-user runs (spawn 10/s, 60 s) in `results/`,
a warm-up of every replica before the clock starts, and the README's two tables
with a written reading of them. The first set was the wrong experiment and the
numbers said so: at 0.5-1.5 s think time, 40 users offer only ~25 req/s, so 1, 2
and 3 containers all landed at 24-27 req/s with the CPU nowhere near its ceiling -
the generator was measuring itself.

**Result at 0.1-0.3 s think time:** 1 container 22.8 req/s at 1700 ms median,
2 containers 32.4 req/s at 780 ms (repeat 23.7), 3 containers 32.9 req/s at 760 ms
once but 26.9 and 22.8 in the other runs, with the only 500s (5 of them,
`sqlite3.OperationalError: database is locked`). Sampled CPU was ~690% of the 800%
available, so on this 8-core laptop a third replica adds contention, not capacity.
The old 20-user CSVs were kept.

**How I changed it:** TODO: (me) reword the "Reading the table" paragraph in my own
voice, and say what I would test next: a fourth container, a GPU host, PostgreSQL.

## Step 9 - One API container, and a stricter think-time setting  (2026-11-26)

**Tool:** Freebuff coding agent.

**Prompt:** "Scale the stack back to one API container and run a one-epoch retrain
to confirm it still works at that size, since that is what I will record. Also
give the locustfile think-time change a last look so a bad value fails clearly
instead of silently."

**What the AI produced:** the stack scaled back to one api container and retrained
twice - 1 epoch finished in 74 s on 2406 clips (0.9883 -> 0.9883) and the 5-epoch
setting used in the video script took 67 s (-> 0.9867), both with a backup. The
`/retrain` guard also proved itself (HTTP 400 with no uploads). In `locustfile.py`,
`think_time_bounds()` now reads the two `LOCUST_WAIT_*` values at load and exits 1
with a clear message for a value that is not a number, NaN/inf, negative, or a
maximum below the minimum; before this, `LOCUST_WAIT_MIN=1o` crashed deep inside
Locust and `MIN=5 MAX=1` silently reversed the two.

**Not verified:** the full demo path through the Streamlit UI, and the fact that
the six clips used here came from `data/test`, which is part of the 600-clip
accuracy set - the "after" number in the real demo will differ.

**How I changed it:** TODO: (me) what I changed in the demo run, e.g. the number of
epochs I picked so the retrain fits the video.
