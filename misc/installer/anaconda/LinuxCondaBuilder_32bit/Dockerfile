FROM castis/centos5-i386

MAINTAINER Lion Krischer

RUN yum -y update || true
RUN yum install curl gcc bzip2 tar

#RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh -o miniconda.sh
RUN curl https://repo.continuum.io/miniconda/Miniconda3-3.19.0-Linux-x86.sh -o miniconda.sh

## Force installation.
RUN yes yes | bash miniconda.sh -b -p /miniconda

RUN /miniconda/bin/conda update --yes conda
RUN /miniconda/bin/conda install --yes conda-build anaconda-client jinja2

RUN mkdir -p /temporary/obspy
COPY meta.yaml /temporary/obspy/meta.yaml

# Tests can fail on occasion. We still want the image to be created.
RUN /miniconda/bin/conda build --py 27 /temporary/obspy
# 3.3 has some dependency conflicts. Maybe they get resolved in the future but
# not that important otherwise.
# RUN /miniconda/bin/conda build --py 33 /temporary/obspy
RUN /miniconda/bin/conda build --py 34 /temporary/obspy
RUN /miniconda/bin/conda build --py 35 /temporary/obspy
RUN /miniconda/bin/conda build --py 36 /temporary/obspy
