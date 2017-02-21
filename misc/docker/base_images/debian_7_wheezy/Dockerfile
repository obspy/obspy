FROM debian:wheezy

MAINTAINER Lion Krischer

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV DEBCONF_NOWARNINGS yes

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
RUN apt-get install -y python python-dev python-setuptools python-support python-numpy lsb-release gcc
RUN apt-get install -y vim git fakeroot equivs lintian help2man quilt git
RUN apt-get install -y ttf-bitstream-vera
RUN apt-get install -y python-matplotlib python-numpy python-scipy python-lxml python-sqlalchemy python-nose python-setuptools python-imaging python-mock python-decorator python-requests python-tornado python-geographiclib
RUN apt-get install -y debhelper python-all devscripts

# install fake future packages, so that we can properly install built obspy deb
# packages to test them (we install future via pip)
RUN cd /tmp && echo "Package: python-future" > python-future.control && equivs-build python-future.control && dpkg -i python-future_*.deb
# debian package for obspy 1.0.2 still have suds listed as dependency
# for now mock install it, can be removed when 1.0.3 is released
RUN cd /tmp && echo "Package: python-suds" > python-suds.control && equivs-build python-suds.control && dpkg -i python-suds_*.deb
