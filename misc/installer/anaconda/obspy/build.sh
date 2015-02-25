#!/bin/bash
if [ -z "$OSX_ARCH" ]; then
$PYTHON setup.py install
else
$PYTHON setup.py install
fi
