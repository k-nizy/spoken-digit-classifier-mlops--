# One image for both services: the API runs uvicorn, the UI runs Streamlit.
# docker-compose.yml overrides the command for each container.
FROM python:3.12-slim

# Byte-code off, unbuffered logs (so `docker compose logs -f` shows output live),
# and quiet TensorFlow C++ chatter.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# libsndfile1 - decoding wav/flac/ogg (and mp3 with libsndfile >= 1.1) for
#                soundfile/librosa; without it every upload fails to load
# libgomp1    - OpenMP runtime that TensorFlow's wheels link against
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first: this layer is cached until requirements.txt changes, so
# editing code does not re-download TensorFlow (the slow part).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Code, the trained model, and the labelled data the retrain endpoint replays.
COPY src ./src
COPY app ./app
COPY models ./models
COPY data/train ./data/train
COPY data/test ./data/test

# Writable places. docker-compose mounts named volumes over uploads/ and db/,
# so uploaded clips and the SQLite file survive `docker compose down`.
RUN mkdir -p data/uploads data/db

ENV MODEL_PATH=/app/models/spoken_digit_cnn.keras \
    DATABASE_PATH=/app/data/db/app.db

EXPOSE 8000 8501

# Default: the API. The UI service replaces this command.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
