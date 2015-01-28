FROM debian:squeeze

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN apt-get update && apt-get upgrade || true
RUN apt-get -y install python-numpy python-scipy gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-pip libgeos-3.2.0 python-dev libgeos-dev ttf-bitstream-vera proj-data
RUN pip install flake8 suds-jurko future
# Squeeze unfortunately has no basemap packages.
RUN GEOS_DIR=/usr/lib pip install basemap
