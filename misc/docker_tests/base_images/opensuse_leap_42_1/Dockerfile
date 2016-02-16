FROM opensuse:leap

MAINTAINER Lion Krischer

RUN zypper --non-interactive addrepo http://download.opensuse.org/repositories/Application:/Geo/openSUSE_Leap_42.1/Application:Geo.repo
RUN zypper --non-interactive --no-gpg-checks refresh
RUN zypper --non-interactive update
RUN zypper --non-interactive install gcc python-devel python-numpy python-scipy python-matplotlib python-SQLAlchemy python-lxml python-mock python-pip python-tornado python-requests python-decorator python-flake8 python-basemap python-basemap-data python-nose
RUN pip install --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.org future
RUN zypper --non-interactive install ca-certificates-mozilla
