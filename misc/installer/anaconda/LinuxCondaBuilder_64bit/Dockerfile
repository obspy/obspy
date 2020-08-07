FROM quay.io/pypa/manylinux1_x86_64

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y gcc tar bzip2

# see below comment, conda>=4.3.27 does not work with CentOS 5 / glibc 2.5, so
# use last conda version that works. Eventually CentOS 5 can be dropped, it is
# already end-of-life. But currently the build still seems to work, so it
# probably doesn't hurt to be nice to ancient Linux 32 bit boxes that might
# still be encountered very sporadically.
#RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh -o miniconda.sh
RUN curl https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86.sh -o miniconda.sh
RUN chmod +x miniconda.sh

RUN ./miniconda.sh -b -p /miniconda

# Starting with conda==4.3.27, Anaconda drops support for CentOS 5 / glibc 2.5
# and instead uses CentOS 6 / glibc 2.12 as minimum baseline.
# So, to build against glibc 2.5 we need to fix the conda version to use.
# See https://github.com/conda/conda/issues/6041#issuecomment-333215057
RUN /miniconda/bin/conda install --yes 'conda<4.3.27'
# Avoid accidentally updating to conda>=4.3.27 later on
RUN /miniconda/bin/conda config --set auto_update_conda False
RUN /miniconda/bin/conda config --set always_yes True
RUN /miniconda/bin/conda install --yes conda-build anaconda-client jinja2

RUN mkdir -p /temporary/obspy
COPY meta.yaml /temporary/obspy/meta.yaml

# Tests can fail on occasion. We still want the image to be created.
RUN /miniconda/bin/conda build --py 27 /temporary/obspy
RUN /miniconda/bin/conda build --py 34 /temporary/obspy
RUN /miniconda/bin/conda build --py 35 /temporary/obspy
RUN /miniconda/bin/conda build --py 36 /temporary/obspy
