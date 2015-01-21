FROM fedora:20

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado python-pip
RUN pip install flake8 suds-jurko future
