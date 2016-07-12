FROM opensuse:13.2

MAINTAINER Lion Krischer

RUN zypper --non-interactive addrepo http://download.opensuse.org/repositories/Application:/Geo/openSUSE_13.2/Application:Geo.repo
RUN zypper --non-interactive --no-gpg-checks refresh
RUN zypper --non-interactive update
RUN zypper --non-interactive install gcc python-devel python-numpy python-scipy python-matplotlib python-SQLAlchemy python-lxml python-mock python-pip python-tornado python-requests python-decorator python-flake8 python-basemap python-basemap-data python-nose ca-certificates-mozilla
RUN pip install future
