FROM debian:jessie

MAINTAINER Lion Krischer

# Set the env variables to non-interactive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV DEBCONF_NOWARNINGS yes

# installation for running tests in docker image
# Can fail on occasion.
RUN apt-get update && apt-get upgrade -y || true
RUN apt-get -y install python-numpy python-scipy python-matplotlib python-sqlalchemy python-lxml python-mock python-mpltoolkits.basemap python-mpltoolkits.basemap-data python-pip python-tornado ttf-bitstream-vera python-decorator python-requests
RUN pip install future
RUN pip install https://github.com/Damgaard/PyImgur/archive/9ebd8bed9b3d0ae2797950876f5c1e64a560f7d8.zip

# additional installation for building deb packages in docker image
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN apt-get install -y python
RUN apt-get install -y python python-dev python-setuptools python-support python-numpy lsb-release gcc
RUN apt-get install -y vim git fakeroot equivs lintian help2man quilt git
RUN apt-get install -y ttf-bitstream-vera
RUN apt-get install -y python-matplotlib python-numpy python-scipy python-lxml python-sqlalchemy python-nose python-flake8 python-setuptools python-imaging python-mock python-decorator python-requests python-tornado python-geographiclib
RUN apt-get install -y python3 python3-dev python3-numpy python3-scipy python3-lxml python3-matplotlib python3-sqlalchemy python3-tornado python3-nose python3-mock python3-flake8 python3-decorator python3-requests python3-tornado python3-pip python3-geographiclib
RUN apt-get install -y debhelper python-all python3-all devscripts
RUN pip3 install future

# install fake future packages, so that we can properly install built obspy deb
# packages to test them (we install future via pip)
RUN cd /tmp && echo "Package: python-future" > python-future.control && equivs-build python-future.control && dpkg -i python-future_*.deb
RUN cd /tmp && echo "Package: python3-future" > python3-future.control && equivs-build python3-future.control && dpkg -i python3-future_*.deb
# debian package for obspy 1.0.2 still have suds listed as dependency
# for now mock install it, can be removed when 1.0.3 is released
RUN cd /tmp && echo "Package: python-suds" > python-suds.control && equivs-build python-suds.control && dpkg -i python-suds_*.deb
RUN cd /tmp && echo "Package: python3-suds" > python3-suds.control && equivs-build python3-suds.control && dpkg -i python3-suds_*.deb
