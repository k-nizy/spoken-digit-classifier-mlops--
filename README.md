# TODO: project title

TODO: one paragraph describing the problem, the dataset and the data type (sound).

## Demo video

TODO: YouTube link (must show camera on, prediction on an uploaded clip, and retraining).

## Deployed application

TODO: public URL of the UI and the API.

## Repository layout

TODO: short map of the folders.

## Setup

TODO: installation steps (venv, requirements, downloading the data, running the notebook).

## Running the API and the UI

TODO: commands (added in steps 3 and 4).

## Docker and scaling

TODO: build/run commands and how to scale the API to 1, 2 and 3 containers.

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

TODO: the 4+ metrics and what they tell you about the model.

## Limitations and next steps

TODO.
