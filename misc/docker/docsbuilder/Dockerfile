FROM python:3.8-slim-buster

LABEL name="obspy-docbuilder"
LABEL description="Image for building docs for obspy"
LABEL maintainer="Damian Kula, dkula@unistra.fr"
LABEL version="1.0.0"
LABEL date="2021.02.10"
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

RUN python -m pip install --upgrade --no-cache-dir pip && \
    python -m pip install --upgrade --no-cache-dir setuptools wheel && \
    python -m pip install --no-cache-dir \
        'numpy>=1.15.0' \
        'scipy>=1.0.0' \
        'matplotlib>=3.2.0' \
        'lxml' \
        'setuptools' \
        'sqlalchemy' \
        'decorator' \
        'requests' \
        'flake8>=2' \
        'pyimgur' \
        'pyproj' \
        'pep8-naming' \
        'cryptography' \
        'pyshp' \
        'sphinx>=3.5.1' \
        'm2r2>=0.2.7' \
        'pybtex>=0.22.2' \
        'sphinx_rtd_theme' \
        'matplotlib' && \
    python -m pip install --no-cache-dir 'cartopy>=0.18.0' && \
    mkdir /obspy

COPY . /obspy

RUN python -m pip install -e /obspy

WORKDIR /obspy/misc/docs

ENTRYPOINT ["make"]
CMD ["html"]
