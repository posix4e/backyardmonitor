FROM python:3.11-slim

# Minimal libs for opencv-python-headless to function
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml /app/
COPY uv.lock /app/
RUN pip install --upgrade pip setuptools wheel

# Copy source
COPY . /app

# Install package in editable mode (brings fastapi, uvicorn, numpy, opencv-python-headless)
RUN pip install -e .

# Defaults; override via env or compose
ENV HOST=0.0.0.0 PORT=8080 DATA_DIR=/data AUTO_START=true

# Persist data outside the container
VOLUME ["/data"]

EXPOSE 8080

CMD ["backyardmonitor", "--host", "0.0.0.0", "--port", "8080", "--env-file", "/app/.env"]

