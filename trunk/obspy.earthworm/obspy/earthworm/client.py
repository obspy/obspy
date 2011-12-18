# -*- coding: utf-8 -*-
"""
Earthworm Wave Server client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU General Public License (GPLv2)
    (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

.. seealso:: http://www.isti2.com/ew/PROGRAMMER/wsv_protocol.html
"""

from obspy.core import Stream
from obspy.earthworm.waveserver import readWaveServerV


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

        See :class:`obspy.earthworm.client.Client` for all parameters.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, cleanup=True):
        """
        Retrieves waveform data from Earthworm Wave Server and returns an ObsPy
        Stream object.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'UW'``.
        :type station: str
        :param station: Station code, e.g. ``'LON'``.
        :type location: str
        :param location: Location code, e.g. ``''``.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.
        :type cleanup: bool
        :param cleanup: Specifies whether perfectly aligned traces should be
            merged or not. See :meth:`~obspy.core.stream.Stream.merge`,
            `method` -1 or :meth:`~obspy.core.stream.Stream._cleanup`.

        .. rubric:: Example

        >>> from obspy.earthworm import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client("hood.ess.washington.edu", 16021)
        >>> dt = UTCDateTime() - 2000  # now - 2000 seconds
        >>> st = client.getWaveform('UW', 'LON', '', 'BHZ', dt, dt + 300)
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy.earthworm import Client
            from obspy.core import UTCDateTime
            client = Client("hood.ess.washington.edu", 16021)
            dt = UTCDateTime() - 2000  # now - 2000 seconds
            st = client.getWaveform('UW', 'LON', '', 'BHZ', dt, dt + 300)
            st.plot()
        """
        if location == '':
            location = '--'
        scnl = (station, channel, network, location)
        # fetch waveform
        tbl = readWaveServerV(self.host, self.port, scnl, starttime, endtime)
        # create new stream
        st = Stream()
        for tb in tbl:
            st.append(tb.getObspyTrace())
        if cleanup:
            st._cleanup()
        return st

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, format="MSEED", cleanup=True):
        """
        Writes a retrieved waveform directly into a file.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'UW'``.
        :type station: str
        :param station: Station code, e.g. ``'LON'``.
        :type location: str
        :param location: Location code, e.g. ``''``.
        :type channel: str
        :param channel: Channel code, e.g. ``'BHZ'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Depending on your ObsPy installation one
            of ``"MSEED"``, ``"GSE2"``, ``"SAC"``, ``"SACXY"``, ``"Q"``,
            ``"SH_ASC"``, ``"SEGY"``, ``"SU"``, ``"WAV"``. See the Supported
            Formats section in method :meth:`~obspy.core.stream.Stream.write`
            for a full list of supported formats. Defaults to ``'MSEED'``.
        :type cleanup: bool
        :param cleanup: Specifies whether perfectly aligned traces should be
            merged or not. See :meth:`~obspy.core.stream.Stream.merge`,
            `method` -1 or :meth:`~obspy.core.stream.Stream._cleanup`.
        :return: None

        .. rubric:: Example

        >>> from obspy.earthworm import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client("hood.ess.washington.edu", 16021)
        >>> t = UTCDateTime() - 2000  # now - 2000 seconds
        >>> client.saveWaveform('UW.LON..BHZ.mseed', 'UW', 'LON', '', 'BHZ',
        ...                     t, t + 300, format='MSEED')  # doctest: +SKIP
        """
        st = self.getWaveform(network, station, location, channel, starttime,
                              endtime, cleanup=cleanup)
        st.write(filename, format=format)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
