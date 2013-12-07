#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USAGE: %prog longitude latitude

Get Flinn-Engahl region name from longitude and latitude
"""
import sys
from optparse import OptionParser
from obspy import __version__
from obspy.core.util import FlinnEngdahl


def main():
    parser = OptionParser(__doc__.strip(), version="%prog " + __version__)
    (_options, args) = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit(1)

    longitude = float(args[0])
    latitude = float(args[1])

    flinn_engdahl = FlinnEngdahl()
    print flinn_engdahl.get_region(longitude, latitude)


if __name__ == '__main__':
    # It is not possible to add the code of main directly to here.
    # This script is automatically installed with name obspy-... by
    # setup.py to the Scripts or bin directory of your Python distribution
    # setup.py needs a function to which it's scripts can be linked.
    main()
