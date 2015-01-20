FROM opensuse:13.2

MAINTAINER Lion Krischer

RUN zypper addrepo http://download.opensuse.org/repositories/Application:/Geo/openSUSE_13.2/Application:Geo.repo
RUN zypper --no-gpg-checks refresh
RUN zypper update
RUN zypper install -y python-devel python-numpy python-scipy python-matplotlib gcc-fortran python-SQLAlchemy python-lxml python-mock python-pip libgeos0 libgeos-devel python-numpy-devel python-tornado
RUN pip install -U setuptools
RUN pip install flake8 suds-jurko future
RUN GEOS_DIR=/usr/lib64 pip install basemap --allow-unverified basemap --allow-external basemap
