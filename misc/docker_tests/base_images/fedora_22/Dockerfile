FROM fedora:22

MAINTAINER Lion Krischer

# Can fail on occasion.
RUN yum -y upgrade || true
RUN yum install -y gcc numpy scipy python-matplotlib python-sqlalchemy python-lxml python-mock python-basemap python-basemap-data python-tornado python-pip python-decorator python-requests
RUN pip install future
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
