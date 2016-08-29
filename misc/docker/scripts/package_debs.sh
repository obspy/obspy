#!/bin/env bash
red='\e[0;31m'
green='\e[0;32m'
no_color='\e[0m'

# Install ObsPy and run the tests.
cd /obspy/misc/debian

# In the following, we're checking some return statuses of commands that are
# piped into `tee`. To avoid testing `tee`'s return status set "pipefail"
# option.
set -o pipefail

./deb__build_debs.sh $1 2>&1 | tee /LOG.txt
if [ $? != 0 ]; then
    echo -e "${red}Deb packaging failed!${no_color}"
    touch /failure
else
    echo -e "${green}Deb packaging successful!${no_color}"
    touch /success
fi

echo "Done with everything!"
