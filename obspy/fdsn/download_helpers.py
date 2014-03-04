#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download helpers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from collections import namedtuple
from obspy.fdsn.header import URL_MAPPINGS


domain = namedtuple("domain", [
    "minimum_latitude",
    "maximum_latitude",
    "minimum_longitude",
    "maximum_longitude"])


def get_availability(starttime, endtime, domain, clients=[]):
    pass
