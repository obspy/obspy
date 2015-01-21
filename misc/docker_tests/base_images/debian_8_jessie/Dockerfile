FROM debian:jessie

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN apt-get update && apt-get upgrade || true
RUN apt-get -y install python-numpy python-scipy gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-mpltoolkits.basemap python-mpltoolkits.basemap-data python-pip python-tornado ttf-bitstream-vera proj-data
RUN pip install flake8 suds-jurko future
