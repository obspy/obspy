FROM quay.io/pypa/manylinux1_x86_64

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y gcc tar bzip2

RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
RUN chmod +x miniconda.sh

RUN ./miniconda.sh -b -p /miniconda

RUN /miniconda/bin/conda update --yes conda
RUN /miniconda/bin/conda install --yes conda-build anaconda-client jinja2

RUN mkdir -p /temporary/obspy
COPY meta.yaml /temporary/obspy/meta.yaml

# Tests can fail on occasion. We still want the image to be created.
RUN /miniconda/bin/conda build --py 27 /temporary/obspy
RUN /miniconda/bin/conda build --py 33 /temporary/obspy
RUN /miniconda/bin/conda build --py 34 /temporary/obspy
RUN /miniconda/bin/conda build --py 35 /temporary/obspy
RUN /miniconda/bin/conda build --py 36 /temporary/obspy
