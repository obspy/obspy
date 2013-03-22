#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the SeismicStation class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.station.util import BaseNode


class SeismicStation(BaseNode):
    """
    From the StationXML definition:
        This type represents a Station epoch. It is common to only have a
        single station epoch with the station's creation and termination dates
        as the epoch start and end dates.
    """
    def __init__(self, *args, **kwargs):
        """
        :param latitude:
        :param longitude:
        :param elevation:
        :param site: These fields describe the location of the station using
            geopolitical entities (country, city, etc.).
        :param vault: Type of vault, e.g. WWSSN, tunnel, transportable array,
            etc
        :param geology: Type of rock and/or geologic formation.
        :param equiment: Equipment used by all channels at a station.
        """
        super(SeismicStation, self).__init__(*args, **kwargs)
