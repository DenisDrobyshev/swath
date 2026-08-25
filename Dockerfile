# CPU image for the inference service. Training happens elsewhere; what ships
# here is a checkpoint plus enough runtime to serve it.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# rasterio ships manylinux wheels with GDAL bundled, so no system GDAL is needed;
# libexpat is still required by the PROJ database it opens at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[geo,service]"

# Mount checkpoints at /models:
#   docker run -p 8000:8000 -v "$PWD/runs:/models" swath
# The entrypoint reads this, so `docker run swath` needs no arguments.
ENV SWATH_CHECKPOINTS=/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

ENTRYPOINT ["swath", "serve", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
CMD []
