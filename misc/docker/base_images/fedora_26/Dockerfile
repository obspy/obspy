FROM fedora:26

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN dnf -y upgrade || true
RUN dnf install -y \
    gcc \
    libmseed \
    libmseed-devel \
    numpy \
    pyshp \
    python-GeographicLib \
    python-basemap \
    python-basemap-data \
    python-decorator \
    python-future \
    python-lxml \
    python-matplotlib \
    python-mock \
    python-pip \
    python-requests \
    python-sqlalchemy \
    python-tornado \
    redhat-rpm-config \
    scipy
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
