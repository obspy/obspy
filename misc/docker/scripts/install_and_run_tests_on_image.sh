#!/bin/env bash
red='\e[0;31m'
green='\e[0;32m'
no_color='\e[0m'

# Install ObsPy and run the tests.
cd /obspy

# In the following, we're checking some return statuses of commands that are
# piped into `tee`. To avoid testing `tee`'s return status set "pipefail"
# option.
set -o pipefail

if [ $(cat /container_name.txt) == "fedora_26" ]; then
    python setup.py --with-system-libmseed develop -v 2>&1 | tee /INSTALL_LOG.txt
else
    pip install -v -e . 2>&1 | tee /INSTALL_LOG.txt
fi

if [ $? != 0 ]; then
    echo -e "${red}Installation failed!${no_color}"
else
    echo -e "${green}Installation successful!${no_color}"
fi

cd

obspy-runtests -r --keep-images --no-flake8 --node=docker-$(cat /container_name.txt) $1 2>&1 | tee /TEST_LOG.txt
if [ $? != 0 ]; then
    echo -e "${red}Tests failed!${no_color}"
    touch /failure
else
    echo -e "${green}Tests successful!${no_color}"
    touch /success
fi

echo "Done with everything!"
