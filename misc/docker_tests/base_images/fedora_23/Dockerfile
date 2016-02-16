FROM fedora:23

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN dnf -y upgrade || true
RUN dnf install -y gcc redhat-rpm-config numpy scipy python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado python-pip python-decorator python-requests python-flake8 python-future
