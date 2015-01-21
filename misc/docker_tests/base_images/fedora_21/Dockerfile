FROM fedora:21

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y numpy scipy gcc-gfortran python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado python-pip
RUN pip install flake8 suds-jurko future
# Fedora has a slightly messed up basemap installation.
RUN mkdir -p /usr/lib64/python2.7/site-packages/mpl_toolkits/basemap/data/
RUN mv /usr/share/basemap/* /usr/lib64/python2.7/site-packages/mpl_toolkits/basemap/data/