FROM centos:centos6

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y gcc tar bzip2

RUN curl http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
RUN chmod +x miniconda.sh

RUN ./miniconda.sh -b -p /miniconda

RUN /miniconda/bin/conda update --yes conda
RUN /miniconda/bin/conda install --yes conda-build binstar jinja2

RUN mkdir -p /temporary/obspy
COPY meta.yaml /temporary/obspy/meta.yaml

RUN /miniconda/bin/conda build --py all /temporary/obspy
