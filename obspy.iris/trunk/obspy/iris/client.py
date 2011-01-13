# -*- coding: utf-8 -*-
"""
IRIS web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core import UTCDateTime, read, Stream
from obspy.core.util import NamedTemporaryFile
from urllib2 import HTTPError
import os
import sys
import urllib
import urllib2


class Client(object):
    """
    IRIS web service request client.

    Examples
    --------

    >>> from obspy.iris import Client
    >>> from obspy.core import UTCDateTime
    >>>
    >>> t = UTCDateTime("2010-02-27T06:30:00.000")
    >>> client = Client()
    >>>
    >>> st = client.getWaveform("IU", "ANMO", "00", "BHZ", t, t + 20)
    >>> print st
    1 Trace(s) in Stream:
    IU.ANMO.00.BHZ | 2010-02-27T06:30:00.019538Z - 2010-02-27T06:30:20.019538Z | 20.0 Hz, 401 samples
    """
    def __init__(self, base_url="http://www.iris.edu/ws",
                 user="", password="", timeout=10):
        self.base_url = base_url
        self.timeout = timeout
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def _fetch(self, url, **params):
        # replace special characters 
        remoteaddr = self.base_url + url + '?' + urllib.urlencode(params)
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            response = urllib2.urlopen(remoteaddr)
        else:
            response = urllib2.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()
        return doc

    def getWaveform(self, network, station, location, channel, starttime,
                     endtime, quality='B'):
        """
        Gets a ObsPy Stream object.

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '00'.
        channel : string
            Channel code, e.g. 'EHE', wildcards are not allowed.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        quality : 'D', 'R', 'Q', 'M' or 'B', optional
            MiniSEED data quality indicator. M and B (default) are treated the
            same and indicate best available. If M or B are selected, the
            output data records will be stamped with a M.

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        url = '/dataselect/query'
        # build up query
        kwargs = {}
        kwargs['net'] = str(network)[0:2]
        kwargs['sta'] = str(station)[0:5]
        if location:
            kwargs['loc'] = str(location)[0:2]
        else:
            kwargs['loc'] = '--'
        kwargs['cha'] = str(channel)[0:3]
        kwargs['start'] = UTCDateTime(starttime - 1).formatIRISWebService()
        kwargs['end'] = UTCDateTime(endtime + 1).formatIRISWebService()
        if str(quality).upper() in ['D', 'R', 'Q', 'M', 'B']:
            kwargs['quality'] = str(quality).upper()
        try:
            data = self._fetch(url, **kwargs)
        except HTTPError:
            raise Exception("No waveform data available")
        # create temporary file for writing data
        tf = NamedTemporaryFile()
        tf.write(data)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        # trim stream
        stream.trim(starttime, endtime)
        return stream


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
