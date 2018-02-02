#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Get the Flinn-Engdahl region name from longitude and latitude.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from argparse import ArgumentParser

from obspy import __version__
from obspy.geodetics import FlinnEngdahl


def main(argv=None):
    parser = ArgumentParser(prog='obspy-flinn-engdahl',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('longitude', type=float,
                        help='Longitude (in degrees) of point. Positive for '
                             'East, negative for West.')
    parser.add_argument('latitude', type=float,
                        help='Latitude (in degrees) of point. Positive for '
                             'North, negative for South.')
    args = parser.parse_args(argv)

    flinn_engdahl = FlinnEngdahl()
    print(flinn_engdahl.get_region(args.longitude, args.latitude))


if __name__ == '__main__':
    # It is not possible to put the code of main directly here.
    # This script is automatically installed with name obspy-... by
    # setup.py to the Scripts or bin directory of your Python distribution.
    # setup.py needs a function to which its scripts can be linked.
    main()
