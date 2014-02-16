#!/bin/bash

if [ -z "$OSX_ARCH" ]; then
    $PYTHON setup.py install
else
    LDFLAGS="-undefined dynamic_lookup -bundle" $PYTHON setup.py install
fi
