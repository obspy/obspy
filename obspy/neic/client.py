# -*- coding: utf-8 -*-
"""
NEIC CWB Query service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & David Ketchum
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from time import sleep
from obspy import UTCDateTime, read, Stream
from obspy.core.util import NamedTemporaryFile
import socket
import traceback
from obspy.neic.util import ascdate, asctime


class Client(object):
    """
    NEIC CWB QueryServer request client for waveform data

    :type host: str, optional
    :param host: The IP address or DNS name of the server
        (default is "137.227.224.97" for cwbpub.cr.usgs.gov)
    :type port: int, optional
    :param port: The port of the QueryServer (default is ``2061``)
    :type timeout: int, optional
    :param timeout: Wait this much time before timeout is raised (python > 2.6,
        default is ``30``)
    :type debug: bool, optional
    :param debug: if ``True``, print debug information (default is ``False``)

    .. rubric:: Example

    >>> from neic import Client
    >>> from obspy import UTCDateTime
    >>> client = Client(debug=True)
    >>> start = UTCDateTime("2013-03-14T06:31:00.000")
    >>> st = client.getWaveform("USISCO BH.00", start, 360.)
    >>> print st
    >>> print "I do not know"
    """
    def __init__(self, host="137.227.224.97", port=2061, timeout=30,
            debug=False):
        """
        Initializes access to a CWB QueryServer
        """
        if debug:
            print "int __init__" + host + "/" + str(port) + " timeout=" + \
                str(timeout)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime):
        """
        Gets a waveform for a specified net, station, location and channel
        from starttime to endtime. The individual elements can contain wildcard
        "?" representing one character, matches of character ranges (e.g.
        channel="BH[Z12]"). All fields are left justified and padded with
        spaces to the required field width if they are too short.  Use
        getWaveformNSCL for seednames specified with regular expressions.

        :type network: str
        :param network: The 2 character network code or regular expression
            (will be padded with spaces to the right to length 2)
        :type station: str
        :param station:  The 5 character station code or regular expression
            (will be padded with spaces to the right to length 5)
        :type location: str
        :param location: The 2 character location code or regular expression
            (will be padded with spaces to the right to length 2)
        :type channel: str
        :param channel:  The 3 character channel code or regular expression
            (lpadded to 3)
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: Stream object with requested data
        """
        seedname = network.ljust(2, " ") + station.ljust(5, " ") + \
            channel.ljust(3, " ") + location.ljust(2, " ")
        return self.getWaveformNSCL(seedname, starttime, endtime - starttime)

    def getWaveformNSCL(self, seedname, starttime, duration):
        """
        Gets a regular expression of channels from a start time for a duration
        in seconds. The regular expression must represent all characters of
        the 12-character NNSSSSSCCCLL pattern e.g. "US.....[BSHE]HZ.." is
        valid, but "US.....[BSHE]H" is not. Complex regular expressions are
        permitted "US.....BHZ..|CU.....[BH]HZ.."

        .. rubric:: Notes

        For detailed information regarding the usage of regular expressions in
        the query, see also the documentation for CWBQuery ("CWBQuery.doc")
        available at ftp://hazards.cr.usgs.gov/CWBQuery/

        :type seedname: str
        :param seedname: The 12 character seedname or 12 character regexp
            matching channels
        :type start: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param start: The starting date/time to get
        :type duration: float
        :param duration: The duration in seconds to get
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: Stream object with requested data
        """
        start = str(UTCDateTime(starttime)).replace("T", " ").replace("Z", "")
        line = "'-dbg' '-s' '%s' '-b' '%s' '-d' '%s'\t" % \
            (seedname, start, duration)
        if self.debug:
            print ascdate() + " " + asctime() + " line=" + line
        success = False
        while not success:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with NamedTemporaryFile() as tf:
                    if self.debug:
                        print ascdate(), asctime(), "connecting temp file", \
                              tf.name
                    s.connect((self.host, self.port))
                    s.setblocking(0)
                    s.send(line)
                    if self.debug:
                        print ascdate(), asctime(), "Connected - start reads"
                    slept = 0
                    maxslept = self.timeout / 0.05
                    totlen = 0
                    while True:
                        try:
                            data = s.recv(102400)
                            if self.debug:
                                print ascdate(), asctime(), "read len", \
                                    str(len(data)), " total", str(totlen)
                            if data.find("EOR") >= 0:
                                if self.debug:
                                    print ascdate(), asctime(), "<EOR> seen"
                                tf.write(data[0:data.find("<EOR>")])
                                totlen += len(data[0:data.find("<EOR>")])
                                tf.seek(0)
                                try:
                                    st = read(tf.name, 'MSEED')
                                except Exception, e:
                                    st = Stream()
                                st.trim(starttime, starttime + duration)
                                s.close()
                                success = True
                                break
                            else:
                                totlen += len(data)
                                tf.write(data)
                                slept = 0
                        except socket.error as e:
                            if slept > maxslept:
                                print ascdate(), asctime(), \
                                    "Timeout on connection", \
                                    "- try to reconnect"
                                slept = 0
                                s.close()
                            sleep(0.05)
                            slept += 1
            except socket.error as e:
                print traceback.format_exc()
                print "CWB QueryServer at " + self.host + "/" + str(self.port)
            except Exception as e:
                print traceback.format_exc()
                print "**** exception found=" + str(e)
        if self.debug:
            print ascdate() + " " + asctime() + " success?  len=" + str(totlen)
        st.merge(-1)
        return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
