FROM fedora:20

MAINTAINER Lion Krischer

RUN rpm -Uvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-2.noarch.rpm
# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado python-nose python-pip
RUN pip install flake8 suds-jurko future pyproj
