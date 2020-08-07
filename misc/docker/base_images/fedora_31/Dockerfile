FROM fedora:31

MAINTAINER Tobias Megies

# Can fail on occasion.
RUN dnf -y upgrade || true
# looks like all "python-" is just links to respective python3 packages
# might have to adapt package names for next release
RUN dnf install -y \
    python-devel \
    gcc \
    libmseed \
    libmseed-devel \
    m2crypto \
    numpy \
    python-GeographicLib \
    python-basemap-data \
    python-cryptography \
    python-decorator \
    python-future \
    python-jsonschema \
    python-lxml \
    python-matplotlib \
    python-mock \
    python-pip \
    python-requests \
    python-sqlalchemy \
    python-tornado \
    python3-basemap \
    python3-pyproj \
    python3-pyshp \
    redhat-rpm-config \
    scipy
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
