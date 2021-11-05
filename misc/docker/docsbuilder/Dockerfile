FROM python:3.8-slim-bullseye

LABEL name="obspy-docbuilder"
LABEL description="Image with all dependencies to build documentation for obspy"
LABEL maintainer="Damian Kula, heavelock@gmail.com"
LABEL version="1.1.0"
LABEL date="2021.11.05"
LABEL schema-version="1.0.0"

RUN apt-get update -yqq && \
    apt-get install -yqq --no-install-recommends \
        build-essential \
        gcc \
        libproj-dev \
        proj-data \
        libgeos-dev \
        proj-bin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /obspy

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --upgrade setuptools wheel && \
    python -m pip install --no-cache-dir -e /obspy  && \
    python -m pip install --no-cache-dir -r /obspy/misc/docs/requirements.txt

WORKDIR /obspy/misc/docs

ENTRYPOINT ["make"]
CMD ["html-docker"]
