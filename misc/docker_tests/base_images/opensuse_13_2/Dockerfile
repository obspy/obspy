FROM opensuse:13.2

MAINTAINER Lion Krischer

RUN zypper --non-interactive addrepo http://download.opensuse.org/repositories/Application:/Geo/openSUSE_13.2/Application:Geo.repo
RUN zypper --non-interactive --no-gpg-checks refresh
RUN zypper --non-interactive update
RUN zypper --non-interactive install gcc python-devel python-numpy python-scipy python-matplotlib gcc-fortran python-SQLAlchemy python-lxml python-mock python-pip geos-devel python-numpy-devel python-tornado python-requests python-decorator python-flake8
RUN pip install --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.org future
RUN GEOS_DIR=/usr/lib64 pip install basemap --allow-unverified basemap --allow-external basemap
