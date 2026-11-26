# AI usage log

Every step where AI was used, in order: the prompt, what it produced, what broke.
Tool for all ten steps: Freebuff coding agent.

**Summary:** AI wrote a first draft of the code; I ran every step, reviewed every
file, wrote all the prose, and changed: TODO: (me) list your changes here - the
settings you re-ran, the bugs you fixed, the text you rewrote.

## Step 1 - Notebook: data, preprocessing, model, evaluation  (2026-11-24)

**Prompt:** "…AI use is allowed only if I disclose it… Step 1: notebook with data
acquisition, preprocessing, model creation, model testing; one optimisation
technique; at least 4 metrics; save the model to models/."

**Produced:** `requirements.txt`, `.gitignore`, a README whose every paragraph is a
`TODO`, and a 20-cell notebook: download FSDD, split it 80/20 stratified by digit,
load at 8 kHz, pad or trim to 1 s, 64-band log-mel scaled to 0..1, a CNN of three
conv blocks with dropout into a 10-way head, early stopping (patience 8, best
weights restored) with a checkpoint and `ReduceLROnPlateau`, then loss, accuracy,
macro and weighted precision, recall, F1, a classification report and a confusion
matrix, and the model saved as `.h5`, `.keras` and `metrics.json`.

**Problem found:** the first run scored 10% and predicted "6" for everything -
batch normalisation stopped the network learning, and the validation split was cut
from class-ordered data, so it held only digits 8 and 9 and early stopping fired at
epoch 9. Without batch norm (keeping dropout) and shuffled before the split it
reached **0.9883** in 34 epochs.

**How I changed it:** TODO: (me) epochs, learning rate, which plots I kept, a bug I
fixed.

## Step 2 - Move the logic into src/  (2026-11-24)

**Prompt:** "Step 2: move the preprocessing, model building and prediction logic
into src/preprocessing.py, src/model.py and src/prediction.py so the API can reuse
them later."

**Produced:** `preprocessing.py` (load, log-mel, dataset and batch helpers, shared
constants), `model.py` (build, train with early stopping and a warm start from an
existing model, save, load, evaluate) and `prediction.py` (cached model, reset
after retraining, predict one or many files). Checked on the same 600 test clips:
accuracy 0.9883, F1 0.9884, loss 0.0495, and a 2-epoch warm start saved a model.

**How I changed it:** TODO: (me) a renamed function, a changed constant, a check I
added.

## Step 3 - FastAPI service with a database and retraining  (2026-11-24)

**Prompt:** "Step 3: a FastAPI app that predicts one uploaded sound, accepts a bulk
upload of multiple files with the results saved to a database, and exposes a
retrain endpoint that warm-starts from my saved model using the uploaded data."

**Produced:** SQLite through the standard library (`predictions`, `uploads`,
`retrain_jobs`) and eight endpoints: `/health`, `/predict`, `/upload` (labels
optional, otherwise from the file name), `/retrain` (warm start, replays
`data/train`, scores before and after, backs up the old weights), `/retrain/{id}`,
`/uploads`, `/metrics`. On a throwaway database: digit 3 at 0.99999, two uploads
storing 10 correct rows, a 25-epoch retrain on 2410 clips (0.9883 -> 0.9850), and a
background job going running -> finished without blocking the API.

**Problem found:** the first retrain failed with "Cannot take the length of shape
with unknown rank" - the worker thread reused the model cached at startup, so it
now loads its own. Fine-tuning uses 3e-4, and a retrained model is only promoted if
it is at most 0.5 accuracy points worse than the serving one.

**How I changed it:** TODO: (me) the minimum number of uploads, an endpoint of my
own, a change to the schema.

## Step 4 - Streamlit UI, and a fix for concurrent /predict requests  (2026-11-24)

**Prompt:** "Step 4: a Streamlit UI showing model up-time, three feature
visualisations, and buttons to bulk-upload clips and trigger retraining… Also fix
the /predict temp-file collision so concurrent requests cannot overwrite each
other's file."

**Produced:** four tabs - Overview (status, uptime, four metrics, latest job,
counters), Feature insights (mean spectrogram per digit, spectral-centroid boxplot,
energy against length, each with a `TODO` caption), Predict, and Upload & retrain
with a polling job. The bug: every upload was saved under its original name in one
folder, so two people sending `recording.wav` could overwrite each other; it now
uses `tempfile.mkstemp()` and deletes the file in a `finally` block, and the bulk
path opens files exclusively (`"xb"`) with a suffix on a clash. 12 simultaneous
requests with the same name all returned digit 7.

**Problem found:** Streamlit's `AppTest` caught the UI reading
`metrics["uploads"]["uploads"]` where the database returns `total`.

**How I changed it:** TODO: (me) my own chart, a different layout, the
interpretation text for the three charts.

## Step 5 - Dockerfile and docker-compose  (2026-11-24)

**Prompt:** "Step 5: the Dockerfile and docker-compose setup to containerise the
API and the UI. Docker is not installed yet, so write it, explain how to build and
run it…"

**Produced:** a `python:3.12-slim` image with `libsndfile1` and `libgomp1` (without
them every upload fails to decode or TensorFlow cannot load), requirements
installed before the code so edits do not re-download TensorFlow, and the model and
data copied in; a `.dockerignore` that keeps the 1 GB `.venv` out of the build
context; a compose file with an `api` on volumes for uploads and SQLite and a `ui`
that waits for it to be healthy and gets `API_URL`, both health checks using
`urllib` because the image has no curl. Docker was still missing, so it was run
against a clean folder instead: health, predict, a 5-clip upload and a 2-epoch
retrain on 2405 clips in 32 s all worked.

**Problem found:** retraining exposed a crash the step-4 check could not see - the
Overview tab read `latest["job_id"]` where the row is keyed `id`, so the page died
as soon as a retrain existed. Fixed to `latest["id"]`.

**How I changed it:** TODO: (me) base image, a volume, a pinned port.

## Step 6 - Locust flood test and an nginx load balancer  (2026-11-24)

**Prompt:** "Step 6: the flood simulation. Write locustfile.py for the API, put a
load balancer in front of it so I can run docker compose up --scale api=1|2|3, and
give me the table I have to fill in with latency at 1, 2 and 3 containers."

**Produced:** a locustfile (clips read once at import so disk I/O is not timed,
weights `/predict` 6 / `/health` 2 / `/metrics` 1 / `/upload` 1 but gated behind
`LOCUST_UPLOAD=1` because it writes files, think time 1-3 s, a warm-up on start); an
nginx balancer with `server api:8000 resolve` and a Docker DNS resolver so
`--scale api=3` is picked up without a restart (needs nginx >= 1.27.3); a compose
file of three services where `api` only `expose`s 8000 (publishing it breaks
`--scale`); and `PRAGMA journal_mode=WAL` in `db.py` so three writers do not block
readers. Without Docker: `locust --list` passed, a 10-user run made 40 requests with
0 failures, the WAL file appeared. `nginx -t` and the table still needed Docker.

**How I changed it:** TODO: (me) number of users, run length, wait time, worker
processes.

## Step 7 - Install Docker and prove the step-6 stack at 3 replicas  (2026-11-25)

**Prompt:** "Before I install Docker, review the new step 6 files… and make sure
nothing will break my first docker compose up --scale api=3. install the docker"

**Produced:** no code changes - the review found nothing that would break a first
scaled start. It installed Docker Desktop 29.9.3 with Compose v5.7.1 (the direct
installer, because `winget` hung) and confirmed `resolver` is allowed inside an
`upstream` block since nginx 1.27.3. Checked at 3 replicas: `nginx -t` passed, five
containers healthy in ~50 s, `/predict` gave digit 4 at 0.9998, 5 uploads all
correct, 9 requests split 10/10/9, scaling 3 -> 1 gave 10/10 with no restart, and a
1-epoch retrain finished in 39 s on 2405 clips (0.9883 -> 0.9833). Then
`docker compose down -v`.

**How I changed it:** TODO: (me) an nginx tag, a bigger body limit, another
`--scale` value.

## Step 8 - Heavier flood test: 40 users, shorter think time  (2026-11-26)

**Prompt:** "My flood-test table shows 3 containers no better than 2, which will get
questioned on camera. Re-run it with more load (about 40 users, shorter think time),
keep the old CSVs, and update the README table and interpretation so the scaling
story is honest and clear."

**Produced:** `LOCUST_WAIT_MIN` / `LOCUST_WAIT_MAX`, new 40-user runs (spawn 10/s,
60 s) in `results/` with the old CSVs kept, a warm-up of each replica, and the
README's two tables plus a reading of them. The first set was the wrong experiment:
at 0.5-1.5 s think time 40 users offer only ~25 req/s, so all three sizes landed at
24-27 req/s and the CPU never neared its ceiling. At 0.1-0.3 s: 1 container 22.8
req/s at 1700 ms median, 2 containers 32.4 req/s at 780 ms (repeat 23.7), 3
containers 32.9 at 760 ms once but 26.9 and 22.8 otherwise, with the only 500s (5,
`database is locked`). Sampled CPU was ~690% of the 800% available, so a third
replica adds contention, not capacity.

**How I changed it:** TODO: (me) reword "Reading the table" in my own voice, and say
what I would test next - a fourth container, a GPU host, PostgreSQL.

## Step 9 - One API container, and a stricter think-time setting  (2026-11-26)

**Prompt:** "Scale the stack back to one API container and run a one-epoch retrain
to confirm it still works at that size… Also give the locustfile think-time change a
last look so a bad value fails clearly instead of silently."

**Produced:** the stack back at one api container, retrained twice - 1 epoch in 74 s
on 2406 clips (0.9883 -> 0.9883) and the 5 epochs used in the video script in 67 s
(-> 0.9867), both with a backup, and `/retrain` returning 400 with no uploads. In
`locustfile.py`, `think_time_bounds()` now exits 1 with a clear message for a value
that is not a number, NaN/inf, negative, or a maximum below the minimum; before,
`LOCUST_WAIT_MIN=1o` crashed inside Locust and `MIN=5 MAX=1` silently reversed the
two.

**Not verified:** the full demo path through the UI, and the six clips here came
from `data/test`, so the "after" number in the real demo will differ.

**How I changed it:** TODO: (me) the number of epochs I picked so the retrain fits
the video.

## Step 10 - Render blueprint for the two services  (2026-11-26)

**Prompt:** "Add a render.yaml blueprint for the two Render services from the
existing Dockerfile so I can deploy without clicking through the dashboard, and
tell me what the free tier will refuse."

**Produced:** `render.yaml` creating `spoken-digit-api` and `spoken-digit-ui` as
free Docker web services from the same `Dockerfile`, so deploying is New ->
Blueprint instead of filling in two sets of dashboard fields. Port 10000 is a
literal number because a Docker command is not run through a shell and would pass
`$PORT` as text; `/health` and `/_stcore/health` are the health checks; TensorFlow
is limited to one thread of each kind for the 0.1 CPU; and the UI's `API_URL` is
the API's public URL, because a free web service cannot receive traffic over the
private network. Validated against Render's published blueprint schema.

**Not verified:** the deployment itself. The free plan gives 512 MB, which is at
the edge of the 434-510 MiB the API used here at idle, so it may be killed for
memory, and a retrain on 0.1 CPU will outlast the UI's 6-minute polling window.

**How I changed it:** TODO: (me) e.g. whether I raised the plan, changed a
service name, or deployed from the dashboard anyway.
