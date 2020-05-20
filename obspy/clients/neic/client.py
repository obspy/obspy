# -*- coding: utf-8 -*-
"""
NEIC CWB Query service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & David Ketchum
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import socket
import traceback
from time import sleep

from obspy import Stream, UTCDateTime, read
from obspy.clients.neic.util import ascdate, asctime
from obspy.clients.httpproxy import get_proxy_tuple, http_proxy_connect


class Client(object):
    """
    NEIC CWB QueryServer request client for waveform data

    :type host: str, optional
    :param host: The IP address or DNS name of the server
        (default is "137.227.224.97" for cwbpub.cr.usgs.gov)
    :type port: int, optional
    :param port: The port of the QueryServer (default is ``2061``)
    :type timeout: int, optional
    :param timeout: Wait this much time before timeout is raised
        (default is ``30``)
    :type debug: bool, optional
    :param debug: if ``True``, print debug information (default is ``False``)

    .. rubric:: Example

    >>> from obspy.clients.neic import Client
    >>> client = Client()
    >>> t = UTCDateTime() - 5 * 3600  # 5 hours before now
    >>> st = client.get_waveforms("IU", "ANMO", "00", "BH?", t, t + 10)
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    >>> st = client.get_waveforms_nscl("IUANMO BH.00", t, 10)
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    IU.ANMO.00.BH... | 40.0 Hz, 401 samples
    """
    def __init__(self, host="137.227.224.97", port=2061, timeout=30,
                 debug=False):
        """
        Initializes access to a CWB QueryServer
        """
        if debug:
            print("int __init__" + host + "/" + str(port) + " timeout=" +
                  str(timeout))
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug
        self.proxy = get_proxy_tuple()

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime):
        """
        Gets a waveform for a specified net, station, location and channel
        from start time to end time. The individual elements can contain
        wildcard ``"?"`` representing one character, matches of character
        ranges (e.g. ``channel="BH[Z12]"``). All fields are left justified and
        padded with spaces to the required field width if they are too short.
        Use get_waveforms_nscl for seednames specified with regular
        expressions.

        .. rubric:: Notes

        Using ``".*"`` regular expression might or might not work. If the 12
        character seed name regular expression is less than 12 characters it
        might get padded with spaces on the server side.

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
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: Stream object with requested data

        .. rubric:: Example

        >>> from obspy.clients.neic import Client
        >>> client = Client()
        >>> t = UTCDateTime() - 5 * 3600  # 5 hours before now
        >>> st = client.get_waveforms("IU", "ANMO", "0?", "BH?", t, t + 10)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        """
        # padding channel with spaces does not make sense
        if len(channel) < 3 and channel != ".*":
            msg = "channel expression matches less than 3 characters " + \
                  "(use e.g. 'BHZ', 'BH?', 'BH[Z12]', 'B??')"
            raise Exception(msg)
        seedname = '%-2s%-5s%s%-2s' % (network, station, channel, location)
        # allow UNIX style "?" wildcard
        seedname = seedname.replace("?", ".")
        return self.get_waveforms_nscl(seedname, starttime,
                                       endtime - starttime)

    def get_waveforms_nscl(self, seedname, starttime, duration):
        """
        Gets a regular expression of channels from a start time for a duration
        in seconds. The regular expression must represent all characters of
        the 12-character NNSSSSSCCCLL pattern e.g. "US.....[BSHE]HZ.." is
        valid, but "US.....[BSHE]H" is not. Complex regular expressions are
        permitted "US.....BHZ..|CU.....[BH]HZ.."

        .. rubric:: Notes

        For detailed information regarding the usage of regular expressions
        in the query, see also the documentation for CWBQuery ("CWBQuery.doc")
        available at ftp://hazards.cr.usgs.gov/CWBQuery/.
        Using ".*" regular expression might or might not work. If the 12
        character seed name regular expression is less than 12 characters it
        might get padded with spaces on the server side.

        :type seedname: str
        :param seedname: The 12 character seedname or 12 character regexp
            matching channels
        :type start: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param start: The starting date/time to get
        :type duration: float
        :param duration: The duration in seconds to get
        :rtype: :class:`~obspy.core.stream.Stream`
        :returns: Stream object with requested data

        .. rubric:: Example

        >>> from obspy.clients.neic import Client
        >>> from obspy import UTCDateTime
        >>> client = Client()
        >>> t = UTCDateTime() - 5 * 3600  # 5 hours before now
        >>> st = client.get_waveforms_nscl("IUANMO BH.00", t, 10)
        >>> print(st)  # doctest: +ELLIPSIS
        3 Trace(s) in Stream:
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        IU.ANMO.00.BH... | 40.0 Hz, 401 samples
        """
        start = str(UTCDateTime(starttime)).replace("T", " ").replace("Z", "")
        line = "'-dbg' '-s' '%s' '-b' '%s' '-d' '%s'\t" % \
            (seedname, start, duration)
        if self.debug:
            print(ascdate() + " " + asctime() + " line=" + line)

        # prepare for routing through http_proxy_connect
        address = (self.host, self.port)
        if self.proxy:
            proxy = (self.proxy.hostname, self.proxy.port)
            auth = ((self.proxy.username, self.proxy.password) if
                    self.proxy.username else None)

        success = False
        while not success:
            try:
                if self.proxy:
                    s, _, _ = http_proxy_connect(address, proxy, auth,
                                                 timeout=self.timeout)
                    # This socket is already connected to the proxy
                else:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    if self.timeout is not None:
                        s.settimeout(self.timeout)
                    s.connect((self.host, self.port))

                with io.BytesIO() as tf:
                    s.send(line.encode('ascii', 'strict'))
                    if self.debug:
                        print(ascdate(), asctime(), "Connected - start reads")
                    slept = 0
                    maxslept = self.timeout / 0.05
                    totlen = 0
                    while True:
                        try:
                            # Recommended bufsize is a small power of 2.
                            data = s.recv(4096)
                            if self.debug:
                                print(ascdate(), asctime(), "read len",
                                      str(len(data)), " total", str(totlen))
                            _pos = data.find(b"<EOR>")
                            # <EOR> can be after every 512 bytes which seems to
                            # be the record length cwb query uses.
                            if _pos >= 0 and (_pos + totlen) % 512 == 0:
                                if self.debug:
                                    print(ascdate(), asctime(), b"<EOR> seen")
                                tf.write(data[0:_pos])
                                totlen += len(data[0:_pos])
                                tf.seek(0)
                                try:
                                    st = read(tf, 'MSEED')
                                except Exception:
                                    st = Stream()
                                st.trim(starttime, starttime + duration)
                                s.close()
                                success = True
                                break
                            else:
                                totlen += len(data)
                                tf.write(data)
                                slept = 0
                        except socket.error:
                            if slept > maxslept:
                                print(ascdate(), asctime(),
                                      "Timeout on connection",
                                      "- try to reconnect")
                                slept = 0
                                s.close()
                            sleep(0.05)
                            slept += 1
            except socket.error:
                print(traceback.format_exc())
                print("CWB QueryServer at " + self.host + "/" + str(self.port))
                raise
            except Exception as e:
                print(traceback.format_exc())
                print("**** exception found=" + str(e))
                raise
        if self.debug:
            print(ascdate() + " " + asctime() + " success?  len=" +
                  str(totlen))
        st.merge(-1)
        return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
