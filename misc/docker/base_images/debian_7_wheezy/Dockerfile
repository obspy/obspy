FROM debian:wheezy

MAINTAINER Tobias Megies

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive DEBIAN_PRIORITY=critical DEBCONF_NOWARNINGS=yes

# install packages to install obspy and build deb packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get -y --no-install-recommends install \
    debhelper \
    devscripts \
    equivs \
    fakeroot \
    gcc \
    git \
    help2man \
    lintian \
    locales \
    lsb-release \
    python \
    python-decorator \
    python-dev \
    python-geographiclib \
    python-imaging \
    python-jsonschema \
    python-lxml \
    python-matplotlib \
    python-m2crypto \
    python-mock \
    python-mpltoolkits.basemap \
    python-mpltoolkits.basemap-data \
    python-nose \
    python-numpy \
    python-pip \
    python-pyproj \
    python-pyshp \
    python-requests \
    python-scipy \
    python-setuptools \
    python-sqlalchemy \
    python-support \
    python-tornado \
    quilt \
    ttf-bitstream-vera \
    vim \
    && rm -rf /var/lib/apt/lists/*
# install some additional packages via pip
RUN pip install future
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
# make sure locale we use in tests is present
RUN locale-gen en_US.UTF-8
RUN echo "backend: agg" > /etc/matplotlibrc

# install fake future packages, so that we can properly install built obspy deb
# packages to test them (we install future via pip)
RUN cd /tmp && echo "Package: python-future" > python-future.control && equivs-build python-future.control && dpkg -i python-future_*.deb
