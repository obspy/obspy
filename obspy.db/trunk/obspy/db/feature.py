# -*- coding: utf-8 -*-
"""
Feature generators for ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import scoreatpercentile


class MinMaxAmplitudeFeature:

    def process(self, trace):
        """
        Gets some statistics about the amplitude values.

        This may take a while to calculate - use a moderate looping interval.

        Example
        -------
        >>> from obspy.core import Trace
        >>> import numpy as np
        >>> tr = Trace(data=np.arange(-5,5))
        >>> result = AmplitudeFeature().process(tr)
        >>> result['max']
        4
        >>> result['upper_quantile']
        1.75
        """
        result = {}
        result['min'] = float(trace.data.min())
        result['max'] = float(trace.data.max())
        result['avg'] = float(trace.data.mean())
        result['median'] = float(scoreatpercentile(trace.data, 50, False))
        result['lower_quantile'] = float(scoreatpercentile(trace.data, 25))
        result['upper_quantile'] = float(scoreatpercentile(trace.data, 75))
        return result


class BWRenamer:

    lookup = {
        'GP01': [('RTPI',
                  UTCDateTime('2009-04-22T06:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP02': [('RTLI',
                  UTCDateTime('2009-04-22T13:00'),
                  UTCDateTime('2009-05-06T09:30'))],
        'GP02': [('RTLI',
                  UTCDateTime('2009-05-06T09:30'),
                  UTCDateTime('2009-05-27T10:00'))],
        'GP10': [('RTLI',
                  UTCDateTime('2009-05-27T09:00'),
                  UTCDateTime('2009-09-29T23:59'))],
        '0001': [('RTLI',
                  UTCDateTime('2009-09-30T00:00'),
                  UTCDateTime('2009-10-02T23:59'))],
        'GP03': [('RTKA',
                  UTCDateTime('2009-05-06T10:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP04': [('RTFA',
                  UTCDateTime('2009-05-06T10:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP05': [('RTSW',
                  UTCDateTime('2009-05-06T10:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP07': [('RTEA',
                  UTCDateTime('2009-05-13T14:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP08': [('RTPA',
                  UTCDateTime('2009-05-10T12:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP09': [('RTSP',
                  UTCDateTime('2009-05-13T12:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP12': [('RTZA',
                  UTCDateTime('2009-06-05T11:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP11': [('RTAK',
                  UTCDateTime('2009-06-04T11:00'),
                  UTCDateTime('2009-12-31T23:59')),
                 ('RTSA',
                  UTCDateTime('2009-05-27T00:17'),
                  UTCDateTime('2009-06-02T12:45'))],
        'GP02': [('RTSL',
                  UTCDateTime('2009-06-30T06:00'),
                  UTCDateTime('2009-12-31T23:59'))],
        'GP13': [('RTSA',
                  UTCDateTime('2009-06-02T13:00'),
                  UTCDateTime('2009-12-11T23:59'))],
        '0002': [('ABRI',
                  UTCDateTime('2009-09-30T00:17'),
                  UTCDateTime('2009-10-02T23:59'))],
    }

    def filter(self, result, trace):
        """
        Filter and rename traces within the BW network.
        """
        # filter only GSE2 files
        if result['format'] != 'GSE2':
            return True
        # only unknown network IDs are allowed
        if result['network'] != '':
            return False
        # set network ID to BW
        result['network'] = 'BW'
        # only specific stations
        if result['station'] not in self.lookup:
            return False
        stations = self.lookup[result['station']]
        for (station, start, end) in stations:
            # check dates
            if trace.stats.starttime < start:
                continue
            if trace.stats.endtime > end:
                continue
            # set new station ID
            result['station'] = station
            # correct channel ID
            channel = result['channel']
            if len(channel) != 3:
                raise TypeError('Channel ID needs 3 chars! Got %s.' % channel)
            result['channel'] = 'EH' + channel[2]
            return result
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
