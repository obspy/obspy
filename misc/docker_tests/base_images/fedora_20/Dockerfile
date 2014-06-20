FROM fedora:20

MAINTAINER Lion Krischer

RUN rpm -Uvh http://ftp.uni-kl.de/pub/linux/fedora-epel/beta/7/x86_64/epel-release-7-0.2.noarch.rpm
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado
RUN easy_install flake8 suds-jurko future
