#!/bin/env bash
red='\e[0;31m'
green='\e[0;32m'
no_color='\e[0m'

# Install ObsPy and run the tests.
cd /

# In the following, we're checking some return statuses of commands that are
# piped into `tee`. To avoid testing `tee`'s return status set "pipefail"
# option.
set -o pipefail

(./deb__build_debs.sh $1 && ls -l /tmp/python-obspy_build/packages/python*-obspy_*.deb /tmp/python-obspy_build/packages/python-obspy-dbg_*.deb) 2>&1 | tee /BUILD_LOG.txt
if [ $? != 0 ]; then
    echo -e "${red}Deb packaging failed!${no_color}"
    touch /build.failure
else
    echo -e "${green}Deb packaging successful!${no_color}"
    touch /build.success
    dpkg -i /tmp/python-obspy_build/packages/python-obspy-dbg_*.deb 2>&1 | tee /TEST_LOG.txt
    dpkg -i /tmp/python-obspy_build/packages/python-obspy_*.deb && apt-get install -f && ((obspy-runtests -r --all --keep-images --no-flake8 --node=docker-deb-$(cat /container_name.txt) 2>&1 | tee --append /TEST_LOG.txt) && touch /test.success || touch /test.failure)
    dpkg -i /tmp/python-obspy_build/packages/python3-obspy_*.deb && apt-get install -f && ((obspy3-runtests -r --all --keep-images --no-flake8 --node=docker-deb-$(cat /container_name.txt) 2>&1 | tee --append /TEST_LOG.txt) && touch /test.success || touch /test.failure)
fi

echo "Done with everything!"
