# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from datetime import datetime
import sys

from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule


def asctime():
    """
    Returns the current time as a string hh:mm:ss
    """
    a = str(datetime.utcnow())
    return a[11:19]


def ascdate():
    """
    Returns the current date at yy/mm/dd
    """
    a = str(datetime.utcnow())
    return a[2:10]


def dsecs(dt):
    """
    Given a timedelta object compute it as double seconds.
    """
    d = dt.days * 86400.
    d = d + dt.seconds
    d = d + dt.microseconds / 1000000.0
    return d


def get_property(filename, key):
    """
    Given a property filename get the value of the given key
    """
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith(key):
            ans = line[len(key) + 1:]
            return ans
    return ""


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={},
    function_map={
        'getProperty': 'obspy.clients.neic.util.get_property'})
