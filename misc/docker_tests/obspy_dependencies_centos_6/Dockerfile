FROM centos:centos6

MAINTAINER Lion Krischer

RUN rpm -Uvh http://download.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy0.8 python-lxml python-mock python-basemap python-basemap-data
RUN easy_install flake8 suds-jurko future
