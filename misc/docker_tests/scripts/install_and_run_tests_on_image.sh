#!/bin/env bash
red='\e[0;31m'
green='\e[0;32m'
no_color='\e[0m'

# Install ObsPy and run the tests.
cd /obspy

pip install -v -e . > /INSTALL_LOG.txt 2>&1

if [ $? != 0 ]; then
    echo -e "${red}Installation failed!${no_color}"
else
    echo -e "${green}Installation successful!${no_color}"
fi

cd

obspy-runtests -r --keep-images --node=docker-$(cat /container_name.txt) > /TEST_LOG.txt 2>&1


if [ $? != 0 ]; then
    echo -e "${red}Tests failed!${no_color}"
else
    echo -e "${green}Tests successful!${no_color}"
fi

echo "Done with everything!"
