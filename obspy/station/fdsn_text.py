#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parsing of the text files from the FDSN station web services.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# The header fields of the text files at the different request levels.
network_components = ("network", "description", "starttime", "endtime",
                      "totalstations")
station_components = ("network", "station", "latitude", "longitude",
                      "elevation", "sitename", "starttime", "endtime")

channel_components = ("network", "station", "location", "channel", "latitude",
                      "longitude", "elevation", "depth", "azimuth", "dip",
                      "instrument", "scale", "scalefreq", "scaleunits",
                      "samplerate", "starttime", "endtime")
all_components = (network_components, station_components, channel_components)


def is_FDSN_station_text_file(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid FDSN
    station text file.

    :param path_or_file_object: File name or file like object.
    """
    try:
        if hasattr(path_or_file_object, "readline"):
            cur_pos = path_or_file_object.tell()
            first_line = path_or_file_object.readline()
        else:
            with open(path_or_file_object, "rt") as fh:
                first_line = fh.readline()
    except:
        return False

    # Attempt to move the file pointer to the old position.
    try:
        path_or_file_object.seek(cur_pos, 0)
    except:
        pass

    first_line = first_line.strip()

    # Attempt to decode.
    try:
        first_line = first_line.decode()
    except:
        pass

    if not first_line.startswith("#"):
        return False
    first_line = first_line.lstrip("#").strip()
    if not first_line:
        return False
    components = tuple(_i.strip().lower() for _i in first_line.split("|"))
    if components in all_components:
        return True
    return False


def read_FDSN_station_text_file(path_or_file_object):
    """
    Function reading a FDSN station text file.

    :param path_or_file_object: File name or file like object.
    """
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
