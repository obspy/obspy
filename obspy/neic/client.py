# -*- coding: utf-8 -*-
"""
NEIC CWB Query service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from time import sleep
from obspy import UTCDateTime, read, Stream
from obspy.core.util import NamedTemporaryFile, _getVersionString
import os
import platform
import socket
import traceback
from neic.util import ascdate, asctime

VERSION = _getVersionString("obspy.neic")
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (VERSION,
    platform.platform(), platform.python_version())


class Client(object):
    """
    NEIC CWB QueryServer request client for waveform data

    :type ipadr: str, optional
    :param ipadr: The IP address or DNS name of the server
        (def=cwbpub.cr.usgs.gov)
    :type port: int, optional
    :param port: The port of the QueryServer (def=2061)
    :type timeout: int, optional (def=30)
    :param timeout: Wait this much time before timeout is raised (python >2.6)
    :type debug: bool, optional
    :param debug: if true, more output is generated (def=False)


    .. rubric:: Example

    >>> from neic import Client
    >>> from obspy import UTCDateTime
    >>> client = Client(debug=True)
    >>> start = UTCDateTime("2013-03-14T06:31:00.000")
    >>> st = client.getWaveForm("USISCO BH.00", start, 360.)
    >>> print st
    >>> print "I do not know"


    """
    def __init__(self, ipadr="137.227.224.97", port=2061, timeout=30,
            debug=False):
        """
        Initializes access to a CWB QueryServer

        See :mod:`obspy.neic` for all parameters.
        """
        if(debug):
            print "int __init__" + ipadr + "/" + str(port) + " timeout=" + \
                str(timeout)
        self.ipadr = ipadr
        self.port = port
        self.timeout = timeout
        self.debug = debug

    def getWaveForm(self, network, station, channel, location, starttime,
            duration):
        """
        getWaveForm() gets a wave for for a separate net, station, channel and
        locaatin for a duration in seconds from a start time.  The individual
        elements can contain wildcard "." representing on space.  All field are
        left justified and padded to the required field width if they are too
        small.  Use getWaveFormNSCL for full regular expressions.

        :type network: str
        :param network:  The 2 character network code or regular expression
            (lpadded to 3)
        :type station: str
        :param station:  The 5 character station code or regular expression
            (lpadded to 5)
        :type channel: str
        :param channel:  The 3 character channel code or regular expression
        (lpad to 3)
        :param location: The 2 character location code or regular expression
        :type start: UTCDateTime
        :param start: The starting date/time to get
        :type duration: float
        :param duration: The duration in seconds to get
        """
        seedname = network.ljust(2, " ") + station.ljust(5, " ") + \
            channel.ljust(3, " ") + location.ljust(2, " ")
        return self.getWaveFormNSCL(seedname, starttime, duration)

    def getWaveFormNSCL(self, seedname, starttime, duration):
        """
        getWaveForm() gets a regular expression of of channels from a start
        time for a duration in seconds.  The regular expression must represent
        all characters of the NNSSSSSCCCLL pattern e.g. US.....[BSHE]HZ.. is
        o.k, but US.....[BSHE]H is not.  Complex regular expressions are
        permitted (US.....BHZ..|CU.....[BH]HZ..)

        :type seedname: str
        :param seedname:  The 12 character seedname or 12 character regexp
            matching channels
        :type start: UTCDateTime
        :param start: The starting date/time to get
        :type duration: float
        :param duration: The duration in seconds to get

        waveforms using the 12 character NNSSSSSCCCLL
        """
        start = str(UTCDateTime(starttime))
        start = start.replace("T", " ")
        start = start.replace("Z", "")
        streams = Stream()
        line = ""
        line += "'-dbg' '-s' '%s' '-b' '%s' '-d' '%s'" % (seedname, start,
            duration)
        if(self.debug):
            print ascdate() + " " + asctime() + " line=" + line
        line += "\t"
        success = False
        while not success:
            try:
                tf = NamedTemporaryFile()
                #print "create socket"
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if(self.debug):
                    print ascdate() + " " + asctime() + \
                        " connecting temp file=" + tf.name
                s.connect((self.ipadr, self.port))
                s.setblocking(0)
                s.send(line)
                if(self.debug):
                    print ascdate() + " " + asctime() + \
                        " Connected - start reads"
                slept = 0
                maxslept = self.timeout / 0.05
                totlen = 0
                while True:
                    try:
                        data = s.recv(102400)
                        if(self.debug):
                            print ascdate() + " " + asctime() + " read len=" \
                                + str(len(data)) + " total=" + str(totlen)
                        if data.find("EOR") >= 0:
                            if(self.debug):
                                print ascdate() + " " + asctime() + \
                                    " <EOR> seen"
                            tf.write(data[0: data.find("<EOR>")])
                            totlen += len(data[0: data.find("<EOR>")])
                            tf.seek(0)
                            try:
                                streams = read(tf.name, 'MSEED')
                            except:
                                streams = Stream()
                            streams.trim(starttime, starttime + duration)
                            s.close()
                            try:
                                if(self.debug):
                                    print ascdate() + " " + asctime() + \
                                        " Delete temp file=" + tf.name
                                os.remove(tf.name)
                            except:
                                print "Failed to remove temp file=" + tf.name
                            success = True
                            break
                        else:
                            totlen += len(data)
                            tf.write(data)
                            slept = 0
                    except socket.error as e:
                        if slept > maxslept:
                            print ascdate() + " " + asctime() + " Timeout " + \
                                "on connection  try to remake connection"
                            slept = 0
                            s.close()
                        sleep(0.05)
                        slept += 1

            except socket.error as e:
                print traceback.format_exc()
                print "CWB QueryServer at " + self.ipadr + "/" + str(self.port)
            except Exception as e:
                print traceback.format_exc()
                print "**** exception found=" + str(e)
        if(self.debug):
            print ascdate() + " " + asctime() + " success?  len=" + str(totlen)
        return streams

if __name__ == '__main__':
    import doctest
    print "In Main"
    doctest.testmod(exclude_empty=True)
