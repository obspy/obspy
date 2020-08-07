# -*- coding: utf-8 -*-
"""
Module to hold a SeedLink stream descriptions (selectors) for network/station.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.utcdatetime import UTCDateTime


class SLNetStation(object):
    """
    Class to hold a SeedLink stream selectors for a network/station.

    :var MAX_SELECTOR_SIZE: Maximum selector size.
    :type MAX_SELECTOR_SIZE: int
    :var net: The network code.
    :type net: str
    :var station: The station code.
    :type station: str
    :var selectors: SeedLink style selectors for this station.
    :type selectors: str
    :var seqnum: SeedLink sequence number of last packet received.
    :type seqnum: int
    :var btime: Time stamp of last packet received.
    :type btime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    """
    MAX_SELECTOR_SIZE = 8

    def __init__(self, net, station, selectors, seqnum, timestamp):
        """
        Creates a new instance of SLNetStation.

        :param net: network code.
        :param station: station code.
        :param selectors: selectors for this net/station, null if none.
        :param seqnum: SeedLink sequence number of last packet received,
            -1 to start at the next data.
        :param timestamp: SeedLink time stamp in a UTCDateTime format for
            last packet received, null for none.
        """
        self.net = str(net)
        self.station = str(station)
        # print "DEBUG: selectors:", selectors
        if selectors is not None:
            self.selectors = selectors
        else:
            self.selectors = []
        self.seqnum = seqnum
        if timestamp is not None:
            self.btime = UTCDateTime(timestamp)
        else:
            self.btime = None

    def append_selectors(self, new_selectors):
        """
        Appends a selectors String to the current selectors for this
        SLNetStation.

        :return: 0 if selectors added successfully, 1 otherwise
        """
        self.selectors.append(new_selectors)
        return 1

    def get_selectors(self):
        """
        Returns the selectors as an array of Strings

        :return: array of selector Strings
        """
        return self.selectors

    def get_sl_time_stamp(self):
        """
        Returns the time stamp in SeedLink string format:
        "year,month,day,hour,minute,second"

        :return: SeedLink time
        """
        return self.btime.format_seedlink()
