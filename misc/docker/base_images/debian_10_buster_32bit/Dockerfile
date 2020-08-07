FROM obspy/base-images:debian_10_buster_32bit

MAINTAINER Tobias Megies

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_PRIORITY=critical
ENV DEBCONF_NOWARNINGS=yes

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
    python-cryptography \
    python-decorator \
    python-dev \
    python-flake8 \
    python-future \
    python-geographiclib \
    python-jsonschema \
    python-lxml \
    python-m2crypto \
    python-matplotlib \
    python-mock \
    python-mpltoolkits.basemap \
    python-mpltoolkits.basemap-data \
    python-nose \
    python-numpy \
    python-pil \
    python-pip \
    python-pyproj \
    python-pyshp \
    python-requests \
    python-scipy \
    python-setuptools \
    python-sqlalchemy \
    python-tornado \
    python-wheel \
    python3 \
    python3-cryptography \
    python3-decorator \
    python3-dev \
    python3-flake8 \
    python3-future \
    python3-geographiclib \
    python3-jsonschema \
    python3-lxml \
    python3-matplotlib \
    python3-mock \
    python3-mpltoolkits.basemap \
    python3-nose \
    python3-numpy \
    python3-pil \
    python3-pip \
    python3-pyproj \
    python3-pyshp \
    python3-requests \
    python3-scipy \
    python3-setuptools \
    python3-sqlalchemy \
    python3-tornado \
    python3-wheel \
    quilt \
    ttf-bitstream-vera \
    vim \
    && rm -rf /var/lib/apt/lists/*
# install some additional packages via pip
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip; \
    pip3 install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
# make sure locale we use in tests is present
RUN locale-gen en_US.UTF-8
