# Spoken Digit Classifier - an end-to-end MLOps pipeline

A CNN that recognises a spoken digit (0-9) from a short audio clip, served through a
FastAPI service with a SQLite database, a Streamlit dashboard, bulk upload, and a
retraining endpoint that fine-tunes the deployed model on newly uploaded clips.

The data is the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset):
3,000 mono WAV recordings at 8 kHz, six speakers (george, jackson, lucas, nicolas,
theo, yweweler) each saying every digit 50 times. The data type is **sound**: each clip
is turned into a 64 x 63 log-mel spectrogram, so an image CNN can classify it.

## Demo video

TODO: YouTube link.

## Deployed application

| Service | URL |
|---|---|
| UI (Streamlit) | https://spoken-digit-ui-p4qz.onrender.com |
| API (FastAPI) | https://spoken-digit-api-p4qz.onrender.com |
| API docs (Swagger) | https://spoken-digit-api-p4qz.onrender.com/docs |

Both run on Render's free plan, so they sleep after 15 minutes without traffic. The
first request after that takes about a minute while the service wakes up.

## Repository layout

```
notebook/spoken_digit_classifier.ipynb   data download, preprocessing, training, evaluation
src/preprocessing.py                     audio -> log-mel spectrogram (shared by everything)
src/model.py                             build, train (with warm start), save, load, evaluate
src/prediction.py                        cached model + prediction helpers
app/main.py                              FastAPI service
app/db.py                                SQLite storage: predictions, uploads, retrain jobs
app/ui.py                                Streamlit dashboard
models/                                  trained model (.keras, .h5), metrics.json, confusion matrix
data/train, data/test                    80/20 split, one folder per digit (2,400 / 600 clips)
locustfile.py, nginx.conf                load test and load balancer
results/                                 Locust CSVs from the flood tests
Dockerfile, docker-compose.yml           containers for local runs and scaling
render.yaml                              Render blueprint for the two cloud services
docs/ai_usage_log.md                     AI usage disclosure
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`data/train` and `data/test` are already in the repository. To rebuild them, run
`notebook/spoken_digit_classifier.ipynb`: it downloads FSDD into `data/raw`, makes a
stratified 80/20 split, trains the CNN and saves it to `models/`.

## Running the API and the UI

Locally, from the repository root:

```bash
python -m uvicorn app.main:app --port 8000          # API -> http://localhost:8000/docs
API_URL=http://localhost:8000 python -m streamlit run app/ui.py    # UI -> http://localhost:8501
```

| Endpoint | What it does |
|---|---|
| `GET /health` | status, model loaded, uptime |
| `POST /predict` | one audio file -> predicted digit and class probabilities |
| `POST /upload` | many files -> predictions stored in the database (label taken from the file name, e.g. `7_george_1.wav`) |
| `GET /uploads` | the stored uploads |
| `POST /retrain` | fine-tune the current model on the labelled uploads (needs at least 5) |
| `GET /retrain/{job_id}` | status and result of a retraining job |
| `GET /metrics` | model metrics and database counts |

The UI has four tabs: **Overview** (uptime and metrics), **Feature insights** (three
charts of the data), **Predict** (one clip), and **Upload & retrain**.

### How retraining works

1. Labelled clips uploaded through `/upload` are saved and recorded in SQLite.
2. `/retrain` starts a background job. It warm-starts from the current model (learning
   rate 3e-4) and trains on the uploads plus replayed clips from `data/train`, so the
   model does not forget what it already knows.
3. The old model is backed up first. Both models are scored on the same 600 test
   clips. If the new one is more than 0.5 accuracy points worse, the old one is
   restored.

On Render's free plan (512 MB RAM) a full replay of all 2,400 training clips runs out
of memory, so `render.yaml` sets `RETRAIN_REPLAY_MAX=200` (a class-balanced random
sample) and `RETRAIN_BATCH_SIZE=16`. A run on the deployed service trained on 205
clips (5 uploaded + 200 replayed) for 5 epochs: accuracy 0.9883 -> 0.9867.

## Docker and scaling

```bash
docker compose up --build                 # api + nginx + ui
# UI:  http://localhost:8501
# API: http://localhost:8000/docs (through nginx)

docker compose up -d --scale api=2        # or 1, or 3
```

nginx on port 8000 round-robins across the `api` replicas. The UI calls the API
through nginx. Uploaded clips and the database live in Docker volumes, so they
survive `docker compose down`.

## Flood simulation with Locust

40 simulated users, spawned at 10/s, for 60 s at each size, against the API through
nginx on `http://127.0.0.1:8000`. Most traffic is `POST /predict`; `GET /health`
and `GET /metrics` make up the rest, and the upload task is off by default
(`LOCUST_UPLOAD=1` turns it on).

The run was repeated at two load levels, because the first level turned out to be
the wrong experiment:

- **Moderate load** - 0.5-1.5 s think time. 40 users only offer about 25 req/s at
  that pace, so every size looks the same; the numbers say more about the load
  generator than about the server.
- **Saturating load** - 0.1-0.3 s think time, same 40 users. Now the requests queue
  up and the server, not the client, decides the result. This is the row that
  answers "does another container help?".

Every replica is warmed up before the clock starts - a freshly started container
serves its first requests from a cold TensorFlow session, which otherwise shows up
as a 502 through nginx and a large `max` value that has nothing to do with scaling.

```bash
docker compose up -d --scale api=1      # then 2, then 3
sleep 50                                 # let the new replica load the model
for i in $(seq 1 40); do curl -s -o /dev/null \
    -F "file=@data/test/4/4_george_1.wav" http://127.0.0.1:8000/predict; done

# saturating run; drop the two LOCUST_WAIT_* variables for the moderate one
LOCUST_WAIT_MIN=0.1 LOCUST_WAIT_MAX=0.3 \
.venv/Scripts/python.exe -m locust -f locustfile.py --host http://127.0.0.1:8000 \
    --headless -u 40 -r 10 -t 60s --csv results/scale1_40u_fast --only-summary
```

**Moderate load (0.5-1.5 s think time)**

| API containers | Requests | Fails | Median (ms) | 95%ile (ms) | Max (ms) | Req/s |
|---|---|---|---|---|---|---|
| 1 | 1425 | 0 | 460 | 1200 | 1542 | 24.4 |
| 2 | 1600 | 0 | 210 | 1200 | 2131 | 27.3 |
| 2 (second run) | 1408 | 1 | 380 | 1500 | 2391 | 24.4 |
| 3 | 1428 | 0 | 330 | 1700 | 5882 | 24.7 |
| 3 (second run) | 1438 | 0 | 340 | 1500 | 3166 | 25.0 |

**Saturating load (0.1-0.3 s think time)**

| API containers | Requests | Fails | Median (ms) | 95%ile (ms) | Max (ms) | Req/s |
|---|---|---|---|---|---|---|
| 1 | 1287 | 0 | 1700 | 2700 | 4446 | 22.8 |
| 2 | 1927 | 0 | 780 | 2500 | 4010 | 32.4 |
| 2 (second run) | 1384 | 0 | 1300 | 3200 | 4549 | 23.7 |
| 3 | 1960 | 0 | 760 | 2700 | 4180 | 32.9 |
| 3 (second run) | 1575 | 1 | 920 | 3400 | 4871 | 26.9 |
| 3 (third run) | 1341 | 5 | 970 | 4100 | 13828 | 22.8 |

Each size was measured two or three times because a single run on a laptop is not
trustworthy; the spread is part of the result. One run is left out: a 2-container run
lost 796 requests when Docker recreated nginx mid-test (a 10 s window of 502s, no
api container actually down). Raw CSVs: `results/scale{1,2,3}_40u*_stats.csv`,
`scale{2a,2b,3b}_40u*_stats.csv`. The earlier lighter 20-user run is still there as
`results/scale{1,2,3}_stats.csv`.

**Reading the table.** A second container clearly helps: under saturating load the
median predict latency drops from 1700 ms to 780 ms and throughput goes from 22.8 to
32.4 req/s (its slower repeat, 23.7 req/s, is the honest error bar). A third
container does not help. Its best run ties the second one (32.9 req/s, 760 ms) and
its other two runs are worse than two containers (26.9 and 22.8 req/s, with the
heaviest tail of the whole experiment), so on this machine 3 is at best a tie and
usually a step backwards. Two measured reasons:

1. **The host runs out of CPU.** All replicas share one 8-core laptop. Sampled
   during a saturating 3-container run, the three api containers used about 230 %
   each - roughly 690 % of the 800 % available - while nginx, the Streamlit UI and
   Locust compete for the rest. Each container is started with
   `TF_NUM_INTRAOP_THREADS=2` (`docker-compose.yml`), so from two replicas on, the
   cores are the limit and the third replica adds contention instead of capacity.
2. **The database is the write bottleneck.** Every `/predict` inserts a row into
   the one SQLite file that all replicas mount. With three writers that file starts
   to lock: the slowest 3-container run logged `sqlite3.OperationalError: database
   is locked` and returned five HTTP 500s, which never happened at one or two
   containers. WAL mode plus `timeout=10` in `app/db.py` makes this rare rather
   than impossible.

So the honest conclusion is: scale out to two replicas for the latency win, do not
expect throughput to grow past the CPU of one host, and move the database off a
shared SQLite file (PostgreSQL, or one writer per replica) before trusting a third
replica. The next measurement worth doing is a distributed Locust run
(`--master` / `--worker`) so the single load-generator process is not part of the
bottleneck.

## Results and interpretation

Evaluated on the 600 held-out test clips (`models/metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | 0.9883 |
| Precision (macro) | 0.9887 |
| Recall (macro) | 0.9883 |
| F1 (macro) | 0.9884 |
| Test loss | 0.0495 |

7 of the 600 clips are misclassified (`models/confusion_matrix.png`). The test set is
balanced (60 clips per digit), so macro and weighted scores are the same. Precision
and recall are almost equal, which means no digit is being over-predicted. The model
stopped early after 34 epochs, with early stopping, dropout and `ReduceLROnPlateau`
as the optimisation techniques.

The split is stratified by digit, not by speaker, so the same six voices appear in
both train and test. The score therefore measures how well the model knows these
speakers, not how it generalises to new ones.

## Limitations and next steps

- **Unseen speakers.** Test on a speaker-held-out split to measure generalisation to
  new voices.
- **No persistent storage on the free plan.** Uploads and the database are lost when a
  Render service restarts. A persistent disk or a hosted database (e.g. PostgreSQL)
  would fix this, and would also remove the SQLite write lock seen at 3 replicas.
- **Retraining inside the API process.** On 512 MB it only fits with a small replay
  sample. A separate worker or a scheduled training job would keep retraining from
  competing with prediction for memory.
- **Clip length.** Clips are padded or trimmed to 1 second, so longer recordings are
  cut off.

## AI usage

An AI coding assistant was used during this project. Every step is recorded in
[`docs/ai_usage_log.md`](docs/ai_usage_log.md).
