#!/usr/bin/env sh
PREFIX=/Applications/ObsPy.app/Contents/MacOS
BIN=$PREFIX/bin

# Uninstall
$BIN/pip uninstall -y obspy

# Leave the current directory, as otherwise the version number magic of ObsPy
# will pickup the git repository and not the one file written in the release
# file. I guess the home directory is a reasonably save choice to not be under
# version control...
pushd $HOME
# Install latest version
$BIN/pip install --no-deps obspy
popd

# Print the current state of the installation
echo ""
echo "$ pip freeze"
echo ""
$BIN/pip freeze
