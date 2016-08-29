FROM edos/wheezy32:latest

MAINTAINER Lion Krischer

# installation for running tests in docker image
# Can fail on occasion.
RUN apt-get update && apt-get upgrade -y || true
RUN apt-get -y install python-numpy python-scipy python-matplotlib python-sqlalchemy python-lxml python-mock python-mpltoolkits.basemap python-mpltoolkits.basemap-data python-pip ttf-bitstream-vera python-decorator python-requests
RUN pip install future
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip
RUN echo "backend: agg" > /etc/matplotlibrc

# additional installation for building deb packages in docker image
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN apt-get install -y python
RUN apt-get install -y python python-dev python-setuptools python-support python-numpy lsb-release gfortran gcc
RUN apt-get install -y vim git fakeroot equivs lintian help2man quilt git
RUN apt-get install -y ttf-bitstream-vera
RUN apt-get install -y python-matplotlib python-numpy python-scipy python-lxml python-suds python-sqlalchemy python-nose python-flake8 python-setuptools python-imaging python-mock python-decorator python-requests python-tornado
RUN apt-get install -y debhelper python-all devscripts
