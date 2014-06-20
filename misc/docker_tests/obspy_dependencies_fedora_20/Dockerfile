FROM fedora:20

MAINTAINER Lion Krischer

RUN rpm -Uvh http://ftp-stud.hs-esslingen.de/pub/epel/beta/7/x86_64/epel-release-7-0.1.noarch.rpm
RUN yum install -y numpy
RUN yum install -y scipy
RUN yum install -y gcc-gfortran
RUN yum install -y python-matplotlib python-sqlalchemy0.8 python-suds python-lxml python-mock
RUN yum install -y python-basemap python-basemap-data
RUN easy_install flake8
