FROM debian:squeeze

MAINTAINER Lion Krischer

RUN apt-get update
RUN apt-get -y install python-numpy python-scipy gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-mpltoolkits.basemap python-mpltoolkits.basemap-data python-pip
RUN pip install flake8 suds-jurko future
