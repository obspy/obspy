# -*- coding: utf-8 -*-
"""
Module of SeedLink utility methods.

Part of Python implementaion of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

class Util(object):
    """ generated source for Util

    """

    @classmethod
    def formatSeedLink(cls, datetime):
        """
        Returns string representation for the SeedLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35.45020)
        >>> Util.formatSeedLink(dt)
        '2008,10,1,12,30,35'

        """

        # round seconds down to integer
        seconds = int(float(datetime.second)
            + float(datetime.microsecond) / 1.0e6)
        return "%d,%d,%d,%d,%d,%g" % (datetime.year, datetime.month, datetime.day,
                                      datetime.hour, datetime.minute,
                                      seconds
                                      )

