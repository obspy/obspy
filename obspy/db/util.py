# -*- coding: utf-8 -*-
"""
Additional utilities for obspy.db.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import UTCDateTime


def parse_mapping_data(lines):
    """
    Parses a mapping file used by the indexer.
    """
    results = {}
    for line in lines:
        if line.startswith('#'):
            continue
        if line.strip() == '':
            continue
        temp = {}
        data = line.split()
        msg = "Invalid format in mapping data: "
        # check old and new ids
        if len(data) < 2 or len(data) > 4:
            raise Exception(msg + 'expected "old_id new_id starttime endtime"')
        elif data[0].count('.') != 3:
            raise Exception(msg + "old id %s must contain 3 dots" % data[0])
        elif data[1].count('.') != 3:
            raise Exception(msg + "new id %s must contain 3 dots" % data[1])
        old_id = data[0]
        n0, s0, l0, c0 = old_id.split('.')
        n1, s1, l1, c1 = data[1].split('.')
        if len(n0) > 2 or len(n1) > 2:
            raise Exception(msg + "network ids must not exceed 2 characters")
        elif len(s0) > 5 or len(s1) > 5:
            raise Exception(msg + "station ids must not exceed 5 characters")
        elif len(l0) > 2 or len(l1) > 2:
            raise Exception(msg + "location ids must not exceed 2 characters")
        elif len(c0) > 3 or len(c1) > 3:
            raise Exception(msg + "channel ids must not exceed 3 characters")
        temp['network'] = n1
        temp['station'] = s1
        temp['location'] = l1
        temp['channel'] = c1
        # check datetimes if any
        if len(data) > 2:
            try:
                temp['starttime'] = UTCDateTime(data[2])
            except Exception:
                msg += "starttime '%s' is not a time format"
                raise Exception(msg % data[2])
        else:
            temp['starttime'] = None
        if len(data) > 3:
            try:
                temp['endtime'] = UTCDateTime(data[3])
            except Exception:
                msg += "endtime '%s' is not a time format"
                raise Exception(msg % data[3])
            if temp['endtime'] < temp['starttime']:
                msg += "endtime '%s' should be after starttime"
                raise Exception(msg % data[3])
        else:
            temp['endtime'] = None
        results.setdefault(old_id, [])
        results.get(old_id).append(temp)
    return results
