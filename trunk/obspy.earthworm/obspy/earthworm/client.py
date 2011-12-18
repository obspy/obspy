# -*- coding: utf-8 -*-
"""
Earthworm WaveServer client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU General Public License (GPLv2)
    (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
"""

from obspy.core import Stream
from obspy.earthworm.waveserver import readWaveServerV


class Client(object):
    """
    A Earthworm WaveServer client.
    """
    def __init__(self, host, port, timeout=None, debug=False):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime):
        """
        Retrieves waveform data from Earthworm WaveServer and returns an ObsPy
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

        .. rubric:: Example

        >>> from obspy.earthworm import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client("hood.ess.washington.edu", 16021)
        >>> dt = UTCDateTime() - 2000  # now - 2000 seconds
        >>> st = client.getWaveform('UW', 'LON', '', 'BHZ', dt, dt + 20)
        >>> st.plot()  # doctest: +SKIP

        .. plot::

            from obspy.earthworm import Client
            from obspy.core import UTCDateTime
            client = Client("hood.ess.washington.edu", 16021)
            dt = UTCDateTime() - 2000  # now - 2000 seconds
            st = client.getWaveform('UW', 'LON', '', 'BHZ', t, t + 20)
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
        return st

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, format="MSEED"):
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
        :type format: ``'MSEED'``, optional
        :param format: Output format. Depending on your ObsPy installation one
            of ``"MSEED"``, ``"GSE2"``, ``"SAC"``, ``"SACXY"``, ``"Q"``,
            ``"SH_ASC"``, ``"SEGY"``, ``"SU"``, ``"WAV"``. See the Supported
            Formats section in method :meth:`~obspy.core.stream.Stream.write`
            for a full list of supported formats.
        :return: None

        .. rubric:: Example

        >>> from obspy.earthworm import Client
        >>> from obspy.core import UTCDateTime
        >>> client = Client("hood.ess.washington.edu", 16021)
        >>> t = UTCDateTime() - 2000  # now - 2000 seconds
        >>> client.saveWaveform('UW.LON..BHZ.mseed', 'UW', 'LON', '', 'BHZ',
        ...                     t, t + 20, format='MSEED')  # doctest: +SKIP
        """
        st = self.getWaveform(network, station, location, channel, starttime,
                              endtime)
        st.write(filename, format=format)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
