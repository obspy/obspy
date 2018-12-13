# -*- coding: utf-8 -*-
"""
Earthworm Wave Server client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. seealso:: http://www.isti2.com/ew/PROGRAMMER/wsv_protocol.html
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from fnmatch import fnmatch

from obspy import Stream, UTCDateTime
from obspy.clients.earthworm.waveserver import get_menu, read_wave_server_v


class Client(object):
    """
    A Earthworm Wave Server client.

    :type host: str
    :param host: Host name of the remote Earthworm Wave Server server.
    :type port: int
    :param port: Port of the remote Earthworm Wave Server server.
    :type timeout: int, optional
    :param timeout: Seconds before a connection timeout is raised (default is
        ``None``).
    :type debug: bool, optional
    :param debug: Enables verbose output of the connection handling (default is
        ``False``).
    """
    def __init__(self, host, port, timeout=None, debug=False):
        """
        Initializes a Earthworm Wave Server client.

        See :class:`obspy.clients.earthworm.client.Client` for all parameters.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, cleanup=True):
        """
        Retrieves waveform data from Earthworm Wave Server and returns an ObsPy
        Stream object.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'UW'``.
        :type station: str
        :param station: Station code, e.g. ``'TUCA'``.
        :type location: str
        :param location: Location code, e.g. ``'--'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``. Last character (i.e.
            component) can be a wildcard ('?' or '*') to fetch `Z`, `N` and
            `E` component.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.
        :type cleanup: bool
        :param cleanup: Specifies whether perfectly aligned traces should be
            merged or not. See :meth:`obspy.core.stream.Stream.merge` for
            ``method=-1``.

        .. rubric:: Example

        >>> from obspy.clients.earthworm import Client
        >>> client = Client("pubavo1.wr.usgs.gov", 16022)
        >>> dt = UTCDateTime() - 2000  # now - 2000 seconds
        >>> st = client.get_waveforms('AV', 'ACH', '', 'BHE', dt, dt + 10)
        >>> st.plot()  # doctest: +SKIP
        >>> st = client.get_waveforms('AV', 'ACH', '', 'BH*', dt, dt + 10)
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy.clients.earthworm import Client
            from obspy import UTCDateTime
            client = Client("pubavo1.wr.usgs.gov", 16022, timeout=5)
            dt = UTCDateTime() - 2000  # now - 2000 seconds
            st = client.get_waveforms('AV', 'ACH', '', 'BHE', dt, dt + 10)
            st.plot()
            st = client.get_waveforms('AV', 'ACH', '', 'BH*', dt, dt + 10)
            st.plot()
        """
        # replace wildcards in last char of channel and fetch all 3 components
        if channel[-1] in "?*":
            st = Stream()
            for comp in ("Z", "N", "E"):
                channel_new = channel[:-1] + comp
                st += self.get_waveforms(network, station, location,
                                         channel_new, starttime, endtime,
                                         cleanup=cleanup)
            return st
        if location == '':
            location = '--'
        scnl = (station, channel, network, location)
        # fetch waveform
        tbl = read_wave_server_v(self.host, self.port, scnl, starttime,
                                 endtime, timeout=self.timeout,
                                 cleanup=cleanup)
        # create new stream
        st = Stream()
        for tb in tbl:
            st.append(tb.get_obspy_trace())
        if cleanup:
            st._cleanup()
        st.trim(starttime, endtime)
        return st

    def save_waveforms(self, filename, network, station, location, channel,
                       starttime, endtime, format="MSEED", cleanup=True):
        """
        Writes a retrieved waveform directly into a file.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'UW'``.
        :type station: str
        :param station: Station code, e.g. ``'TUCA'``.
        :type location: str
        :param location: Location code, e.g. ``''``.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``. Last character (i.e.
            component) can be a wildcard ('?' or '*') to fetch `Z`, `N` and
            `E` component.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. One of ``"MSEED"``, ``"GSE2"``,
            ``"SAC"``, ``"SACXY"``, ``"Q"``, ``"SH_ASC"``, ``"SEGY"``,
            ``"SU"``, ``"WAV"``. See the Supported Formats section in method
            :meth:`~obspy.core.stream.Stream.write` for a full list of
            supported formats. Defaults to ``'MSEED'``.
        :type cleanup: bool
        :param cleanup: Specifies whether perfectly aligned traces should be
            merged or not. See :meth:`~obspy.core.stream.Stream.merge`,
            `method` -1 or :meth:`~obspy.core.stream.Stream._cleanup`.
        :return: None

        .. rubric:: Example

        >>> from obspy.clients.earthworm import Client
        >>> client = Client("pubavo1.wr.usgs.gov", 16022)
        >>> t = UTCDateTime() - 2000  # now - 2000 seconds
        >>> client.save_waveforms('AV.ACH.--.BHE.mseed',
        ...                       'AV', 'ACH', '', 'BHE',
        ...                       t, t + 10, format='MSEED')  # doctest: +SKIP
        """
        st = self.get_waveforms(network, station, location, channel, starttime,
                                endtime, cleanup=cleanup)
        st.write(filename, format=format)

    def get_availability(self, network="*", station="*", location="*",
                         channel="*"):
        """
        Gets a list of data available on the server.

        This method returns information about what time series data is
        available on the server. The query can optionally be restricted to
        specific network, station, channel and/or location criteria.

        :type network: str
        :param network: Network code, e.g. ``'UW'``, wildcards allowed.
        :type station: str
        :param station: Station code, e.g. ``'TUCA'``, wildcards allowed.
        :type location: str
        :param location: Location code, e.g. ``'--'``, wildcards allowed.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``, wildcards allowed.
        :rtype: list
        :return: List of tuples with information on the available data. One
            tuple consists of network, station, location, channel
            (all strings), start time and end time
            (both as :class:`~obspy.core.utcdatetime.UTCDateTime`).

        .. rubric:: Example

        >>> from obspy.clients.earthworm import Client
        >>> client = Client("pubavo1.wr.usgs.gov", 16022, timeout=5)
        >>> response = client.get_availability(
        ...     network="AV", station="ACH", channel="BH*")
        >>> print(response)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [('AV',
          'ACH',
          '--',
          'BHE',
          UTCDateTime(...),
          UTCDateTime(...)),
         ('AV',
          'ACH',
          '--',
          'BHN',
          UTCDateTime(...),
          UTCDateTime(...)),
         ('AV',
          'ACH',
          '--',
          'BHZ',
          UTCDateTime(...),
          UTCDateTime(...))]
        """
        # build up possibly wildcarded trace id pattern for query
        if location == '':
            location = '--'
        pattern = ".".join((network, station, location, channel))
        # get overview of all available data, winston wave servers can not
        # restrict the query via network, station etc. so we do that manually
        response = get_menu(self.host, self.port, timeout=self.timeout)
        # reorder items and convert time info to UTCDateTime
        response = [(x[3], x[1], x[4], x[2], UTCDateTime(x[5]),
                     UTCDateTime(x[6])) for x in response]
        # restrict results according to user input
        response = [x for x in response if fnmatch(".".join(x[:4]), pattern)]
        return response


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
