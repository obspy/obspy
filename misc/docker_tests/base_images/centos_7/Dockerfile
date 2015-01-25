FROM centos:centos7

MAINTAINER Lion Krischer

RUN rpm -Uvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-5.noarch.rpm
# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy0.8 python-lxml python-mock python-basemap python-basemap-data python-pip
RUN easy_install -U setuptools
RUN pip install flake8 suds-jurko future
# CentOS 7 has a slightly messed up basemap installation.
RUN mkdir -p /usr/lib64/python2.7/site-packages/mpl_toolkits/basemap/data/
RUN mv /usr/share/basemap/* /usr/lib64/python2.7/site-packages/mpl_toolkits/basemap/data/
