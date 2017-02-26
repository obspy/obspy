FROM obspy/base-images:ubuntu_14_04_trusty_32bit

MAINTAINER Tobias Megies

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive DEBIAN_PRIORITY=critical DEBCONF_NOWARNINGS=yes

# install packages to install obspy and build deb packages
RUN apt-get update && apt-get upgrade -y || true
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
    python-flake8 \
    python-geographiclib \
    python-imaging \
    python-lxml \
    python-matplotlib \
    python-mock \
    python-mpltoolkits.basemap \
    python-mpltoolkits.basemap-data \
    python-nose \
    python-numpy \
    python-pip \
    python-requests \
    python-scipy \
    python-setuptools \
    python-sqlalchemy \
    python-tornado \
    python3 \
    python3-decorator \
    python3-dev \
    python3-flake8 \
    python3-lxml \
    python3-matplotlib \
    python3-mock \
    python3-nose \
    python3-numpy \
    python3-pip \
    python3-requests \
    python3-scipy \
    python3-sqlalchemy \
    python3-tornado \
    quilt \
    ttf-bitstream-vera \
    vim \
    && rm -rf /var/lib/apt/lists/*
# install some additional packages via pip
RUN pip install future; \
    pip3 install future
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip; \
    pip3 install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip || true
# make sure locale we use in tests is present
RUN locale-gen en_US.UTF-8

# install fake future packages, so that we can properly install built obspy deb
# packages to test them (we install future via pip)
RUN cd /tmp && \
    (echo "Package: python-future" > python-future.control && equivs-build python-future.control && dpkg -i python-future_*.deb); \
    (echo "Package: python3-future" > python3-future.control && equivs-build python3-future.control && dpkg -i python3-future_*.deb)
# debian package for obspy 1.0.2 still have suds listed as dependency
# for now mock install it, can be removed when 1.0.3 is released
RUN cd /tmp && echo "Package: python-suds" > python-suds.control && equivs-build python-suds.control && dpkg -i python-suds_*.deb
RUN cd /tmp && echo "Package: python3-suds" > python3-suds.control && equivs-build python3-suds.control && dpkg -i python3-suds_*.deb
