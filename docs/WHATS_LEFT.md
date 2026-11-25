# What is left to do

Audit of this repository at the end of step 6. Everything the code can do is
already written and tested; this file lists **only** the work that cannot be
done from here, because it needs your machine, your accounts, your voice or
your own words.

Read the "Is this ready to submit?" table at the bottom first if you only have
five minutes.

---

## 1. What is already finished

| Area | Where | Evidence |
|---|---|---|
| Notebook: FSDD download, split, log-mel preprocessing, CNN, training, evaluation, saving | `notebook/spoken_digit_classifier.ipynb` | 25 cells executed, no errors; accuracy **0.9883**, F1 macro **0.9884** |
| Reusable code | `src/preprocessing.py`, `src/model.py`, `src/prediction.py` | imported by the notebook, the API and the UI |
| Trained model + metrics + confusion matrix | `models/spoken_digit_cnn.h5`, `.keras`, `metrics.json`, `confusion_matrix.png` | 600 test clips, 7 errors |
| API with database and retraining | `app/main.py`, `app/db.py` | 8 endpoints verified: `/health`, `/predict`, `/upload`, `/uploads`, `/retrain`, `/retrain/{id}`, `/metrics`, `/docs` |
| Data insights UI | `app/ui.py` | 4 tabs (Overview, Feature insights, Predict, Upload & retrain) |
| Container setup | `Dockerfile`, `.dockerignore`, `docker-compose.yml` | image built (`spoken-digit-api:local`, 1.07 GB) and run as **3 api containers + nginx + ui**: all healthy, predict/upload/metrics/retrain verified through the balancer |
| Load test | `locustfile.py` | ran headless, 0 % failures, numbers in `docs/ai_usage_log.md` step 6 |
| Load balancer | `nginx.conf` (new `nginx` service) | `nginx -t` passes on `nginx:1.29-alpine`; requests split 10/10/9 across the 3 replicas; scaling 3 -> 1 gave 10/10 successes with no nginx restart |
| AI use disclosed | `docs/ai_usage_log.md` | 7 entries, one per step |

---

## 2. The work that is genuinely yours

### 2.1 Docker is installed and the stack is already built  (done - no work left)

Docker Desktop **29.9.3** (Compose **v5.7.1**) is installed and running, and the
image `spoken-digit-api:local` is already built, so the first start downloads
nothing. Just start it:

```bash
cd spoken-digit-classifier-mlops
docker compose up -d --scale api=3
```

Expect five containers, all healthy within about a minute: `api-1`, `api-2`,
`api-3`, `nginx`, `ui`. Then open <http://localhost:8501> (UI) and
<http://localhost:8000/docs> (API). Stop it with `docker compose down` (add
`-v` to also wipe the database and the uploaded clips).

Only `docker compose up --build` is slow, and only if you change
`requirements.txt`: this connection gets roughly **190 kB/s** from PyPI, so the
572 MB TensorFlow wheel alone takes about 45-60 minutes. Nothing is broken, it
is just the download.

What was verified here on 2026-11-25 with 3 API containers:

| Check | Result |
|---|---|
| `docker compose config --quiet` | valid; `--dry-run up --scale api=3 -d` plans api-1/2/3, then nginx, then ui |
| `nginx -t` on `nginx:1.29-alpine` | syntax OK |
| nginx starts with **no** api container | stays up, `/lb-health` 200, upstream 502 (it does not wait for DNS) |
| `docker compose up -d --scale api=3` | 5 containers healthy in ~50 s |
| `/`, `/health`, `/metrics`, `/uploads` through nginx | 200 |
| `/predict` through nginx | digit 4, confidence 0.9998 |
| `/upload` 5 labelled clips through nginx | all 5 correct, no shared-database errors |
| 9 `/health` requests | api-1: 10, api-2: 10, api-3: 9 (even round-robin) |
| `/retrain` with 1 epoch in a container | finished in 39 s on 2405 clips, 0.9883 -> 0.9833 |
| three containers writing one SQLite file | 600 rows, `journal_mode = wal`, no "database is locked" |

One thing to remember for the video: retraining writes the model **inside the
container that handled the request**, so at 3 replicas the other two keep
serving the old weights until you restart them. Do the retraining part of the
demo at `--scale api=1`.

If anything does fail, the useful output is `docker compose logs api --tail 50`.

### 2.2 Produce the flood-test table  (done on 2026-11-26 - re-run only if you change the code)

Done. The README now has two tables (moderate load and saturating load), several
runs per size, and a written reading of them. Nothing to run before the video; if
you do re-run it, the exact protocol (spawn rate, think time, warm-up) is in the
README section "Flood simulation with Locust".

The headline for the video is the saturating table: 1 container 22.8 req/s and
1700 ms median, 2 containers 32.4 req/s (repeat 23.7) and 780 ms, 3 containers at
best 32.9 and in two of three runs worse (26.9, 22.8) with the only 500s.

What was measured, for reference:

| API containers | Requests | Fails | Median (ms) | 95%ile (ms) | Max (ms) | Req/s |
|---|---|---|---|---|---|---|
| 1 (saturating) | 1287 | 0 | 1700 | 2700 | 4446 | 22.8 |
| 2 (saturating) | 1927 | 0 | 780 | 2500 | 4010 | 32.4 |
| 2 (saturating, repeat) | 1384 | 0 | 1300 | 3200 | 4549 | 23.7 |
| 3 (saturating) | 1960 | 0 | 760 | 2700 | 4180 | 32.9 |
| 3 (saturating, repeat) | 1575 | 1 | 920 | 3400 | 4871 | 26.9 |
| 3 (saturating, third run) | 1341 | 5 | 970 | 4100 | 13828 | 22.8 |

`--force-recreate nginx` is **not** needed: `server api:8000 resolve` makes
nginx re-read Docker's DNS every 10 s, and a scale change 3 -> 1 was measured
with 10/10 replies and no nginx restart. The one thing that *does* matter: warm
a newly added replica with a few `/predict` calls before starting the clock,
otherwise the first requests hit a cold TensorFlow session (one 502 and a ~6 s
max in the first attempt).

If you are asked on camera why 3 is not better than 2, you now have two measured
reasons: (1) the laptop has 8 logical cores and the three api containers together
used about 690 % of the 800 % available, so the host is saturated (each container
runs TensorFlow with 2 intraop threads, `docker-compose.yml`); (2) all replicas
write to the same SQLite file and at 3 replicas it starts to lock - the slowest
run logged `sqlite3.OperationalError: database is locked` and returned five HTTP
500s, which never happened with one or two containers.

### 2.3 Deploy it  (about 20 minutes, needs your account)

The app is Python + TensorFlow + SQLite, so it needs a container host, not a
static host. Render is the shortest path because it builds the existing
`Dockerfile` unchanged:

1. Push the repo to GitHub first (section 2.5).
2. Render → New → Web Service → your repo → Language: **Docker**.
   - Name `spoken-digit-api`, Instance type: free.
   - Docker Command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add a Disk (1 GB) mounted at `/app/data/db` if you want the SQLite file to
     survive restarts; without it the database resets on every deploy.
3. Render → New → Web Service → same repo again, this time the UI.
   - Name `spoken-digit-ui`.
   - Docker Command:
     `python -m streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --browser.gatherUsageStats false`
   - Environment variable `API_URL` = the first service's URL, e.g.
     `https://spoken-digit-api.onrender.com`
4. Open the UI URL once, wait for the free instance to wake (30-60 s), predict
   one clip. Then put both URLs in the README.

Free instances sleep after ~15 minutes idle, so wake the API *before* recording
the video. If Render refuses the disk on the free plan, say so in the README and
deploy without it: the model still loads from the image, only the upload history
resets.

### 2.4 Record the demo video  (5-10 minutes)

The rubric wants camera on, the prediction, and the retraining, in one take:

0. **Before you record:** replace the three `TODO: (me)` captions in
   `app/ui.py` (the Feature insights tab, around lines 196, 200 and 204). They
   render on screen as captions, so leave them and the video shows your TODOs.
   Also reset the demo data with `docker compose down -v` so the counters start
   at zero and the audience sees them grow.
1. Camera on, say your name and what you built.
2. Open the deployed UI (or `localhost:8501` if the deployment is asleep).
3. Overview tab: point at the four metrics.
4. Feature insights tab: the three charts.
5. Predict tab: upload a clip from `data/test/` — say the true digit before
   uploading, then show the prediction and the confidence.
6. Upload & retrain tab: upload 5-10 labelled clips, start retraining, show
   accuracy before → after and the job message.
7. Show the row in the database growing (`/uploads` or the tab's table).
8. Upload to YouTube (unlisted is fine) and put the link in the README.

### 2.5 Make the repository public

There is no git repository in this folder yet — only `.gitignore`. That is
deliberate: `.gitignore` already excludes `.venv/`, `data/raw/`, `*.db`,
`__pycache__/`, so the first commit will be clean (~15 MB, the wav files).

```bash
git init
git add .
git status        # confirm .venv, *.db and data/raw are NOT staged
git commit -m "Spoken digit classifier: notebook, API, UI, Docker, load test"
```

Then create an empty repo on GitHub and:

```bash
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

---

## 3. The writing that is yours (by your own rule)

You asked that all explanation text be left as `TODO` so you can write it. These
are the remaining `TODO`s, with the facts to use so you are not starting cold.

### 3.1 README.md — 11 sections

| Section | The numbers/facts to put in it |
|---|---|
| Title | e.g. "Spoken digit classifier — an end-to-end MLOps pipeline" |
| Intro paragraph | FSDD, 3000 recordings of the digits 0-9 by 6 speakers, 8 kHz mono wav; **non-tabular: audio**, turned into 64x63 log-mel spectrograms |
| Demo video | YouTube URL |
| Deployed application | the two Render URLs |
| Repository layout | `notebook/` experiment, `src/` reusable code, `app/` API + UI, `models/` artefacts, `data/` split clips, `docs/ai_usage_log.md`, `locustfile.py`, `nginx.conf` |
| Setup | `python -m venv .venv`, activate, `pip install -r requirements.txt`, run the notebook (it downloads FSDD into `data/raw/` and writes `data/train` + `data/test`) |
| Running the API and the UI | `python -m uvicorn app.main:app --port 8000`, `python -m streamlit run app/ui.py` |
| Docker and scaling | `docker compose up --build`, then `--scale api=3`; nginx on port 8000 round-robins |
| Flood simulation | the commands from 2.2 plus the table you fill in |
| Results and interpretation | accuracy 0.9883, precision macro 0.9887, recall macro 0.9883, F1 macro 0.9884, loss 0.0495, 34 epochs, 7 errors in 600 clips; which digits get confused is in `models/confusion_matrix.png` |
| Limitations and next steps | see section 5 below — you can reuse it in your own words |

### 3.2 Notebook — 10 markdown cells

Cells 0, 3, 5, 8, 10, 13, 15, 18, 22, 24, each with a `TODO:` line saying
exactly what to write.

Facts you will need: **2400 train / 600 test** (80/20, stratified by digit);
padding or trimming to a fixed length; mel filterbank + dB + 0..1
normalisation; three convolutional blocks with dropout then a dense head;
early stopping with **patience 8**, `restore_best_weights=True`,
`ModelCheckpoint` and `ReduceLROnPlateau`, **34 epochs** actually run;
best epoch's accuracy 0.9883 on the 600 held-out clips.

### 3.3 `docs/ai_usage_log.md` — the "How I changed it" lines

Seven `TODO: (me)` lines, one per step (grep for `TODO: (me)`). Each asks what
*you* changed after the AI produced the code, which is the disclosure the
assignment asks for. One or two honest sentences each is enough — e.g. "I
raised the epoch count to 25 for the video retrain because 8 finished too fast
to film."

---

## 4. Is this ready to submit?

Once the items in sections 2 and 3 above are done, yes. Mapping to the rubric:

| Requirement | Status |
|---|---|
| Non-tabular data (audio, not a table) | Done — FSDD wav clips -> log-mel spectrograms |
| Notebook with data acquisition, preprocessing, model, training, evaluation | Done — needs your markdown prose |
| An optimisation technique, explained | Done — early stopping (+ checkpoint, LR schedule) |
| At least 4 evaluation metrics | Done — accuracy, loss, precision, recall, F1 (macro and weighted), per-class report |
| Saved model reused later (not retrained from scratch) | Done — `models/spoken_digit_cnn.keras`, the notebook writes it, the API loads it, retraining warm-starts from it |
| API | Done — 8 endpoints, verified |
| Uploads saved to a database | Done — SQLite: `predictions`, `uploads`, `retrain_jobs`; verified with real uploads |
| UI with data insights | Done — 4 tabs, verified |
| Docker | Done — image built and run as 3 api replicas + nginx + ui; predict, upload, metrics and a real retrain all verified through the balancer |
| Locust flood test at 1/2/3 containers | Ready to run — script verified; the 1/2/3 numbers are yours to collect (item 2.2) |
| Deployed app | **Not deployed** — needs your account, item 2.3 |
| Video with camera on, prediction and retraining | **Not recorded** — item 2.4 |
| AI use disclosed | Done — 7 log entries; your "How I changed it" lines pending |

Nothing else is missing. There is no unfinished code path, no endpoint without
a test, and no placeholder in `app/`, `src/` or `locustfile.py`.

---

## 5. Honest limitations (worth a sentence in the README)

- **The database is SQLite in a shared volume.** Fine for a demo and for the
  flood test (WAL mode is on, so concurrent readers do not block the writer),
  but three API containers writing one SQLite file is not what you would ship;
  a real deployment wants PostgreSQL.
- **Retraining is a single-container operation.** The retrained model is written
  inside the container that received the request, so with `--scale api=3` the
  other replicas keep the old model until recreated, and a fresh image always
  starts from the notebook-trained model. Do the retraining demo with the
  default single `api` container.
- **Retraining runs in a background thread of the API process.** Restarting the
  container mid-job leaves the job row marked `running`; there is no queue.
- **The image is ~3 GB** because `requirements.txt` is shared with the notebook
  (Jupyter, matplotlib). In a real project the serving image would get its own
  slim requirements file.
- **Free hosting sleeps.** First request after idle takes 30-60 s; record the
  video after waking both services.
