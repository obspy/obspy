FROM toopher/centos-i386:centos6

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN linux32 yum install -y gcc tar bzip2


RUN linux32 curl http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh -o miniconda.sh
RUN linux32 chmod +x miniconda.sh

RUN linux32 ./miniconda.sh -b -p /miniconda

RUN linux32 /miniconda/bin/conda update --yes conda
RUN linux32 /miniconda/bin/conda install --yes conda-build binstar jinja2

RUN linux32 mkdir -p /temporary/obspy
COPY meta.yaml /temporary/obspy/meta.yaml

# Tests can fail on occasion. We still want the image to be created.
RUN linux32 /miniconda/bin/conda build --py 26 /temporary/obspy || true
RUN linux32 /miniconda/bin/conda build --py 27 /temporary/obspy || true
RUN linux32 /miniconda/bin/conda build --py 33 /temporary/obspy || true
RUN linux32 /miniconda/bin/conda build --py 34 /temporary/obspy || true
