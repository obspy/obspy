# -*- coding: utf-8 -*-
"""
Module to manage a connection to a SeedLink server using a Socket.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


import time

from obspy.core.utcdatetime import UTCDateTime
from obspy.seedlink.client.slnetstation import SLNetStation
from obspy.seedlink.client.slstate import SLState
from obspy.seedlink.seedlinkexception import SeedLinkException
from obspy.seedlink.sllog import SLLog
from obspy.seedlink.slpacket import SLPacket
import select
import socket


class SeedLinkConnection(object):
    """
    Class to manage a connection to a SeedLink server using a Socket.

    See obspy.realtime.seedlink.SLClient for an example of how to create
    and use this SeedLinkConnection object.
    A new SeedLink application can be created by sub-classing SLClient,
    or by creating a new class and invoking the methods of SeedLinkConnection.

    :var SEEDLINK_PROTOCOL_PREFIX: URI/URL prefix for seedlink
        servers ("seedlnk://").
    :type SEEDLINK_PROTOCOL_PREFIX: str
    :var UNISTATION: The station code used for uni-station mode.
    :type UNISTATION: str
    :var UNINETWORK: The network code used for uni-station mode.
    :type UNINETWORK: str
    :var DFT_READBUF_SIZE: Default size for buffer to hold responses
        from server (default is 1024).
    :type DFT_READBUF_SIZE: int
    :var QUOTE_CHAR: Character used for delimiting timestamp strings in the
        statefile.
    :type QUOTE_CHAR: str

    Publicly accessible (get/set) parameters:

    :var sladdr: The host:port of the SeedLink server.
    :type sladdr: str
    :var keepalive: Interval to send keepalive/heartbeat (seconds)
        (default is 0 sec).
    :type keepalive: int
    :var netto: Network timeout (seconds) (default is 120 sec).
    :type netto: int
    :var netdly: Network reconnect delay (seconds)  (default is 30 sec).
    :type netdly: int
    :var sllog: Logging object (default is sys.stdout).
    :type sllog: :class:`~obspy.seedlink.SLLog`
    :var info_string: String containing concatination of contents of last
        terminated set of INFO packets.
    :type info_string: str
    :var statefile: File name for storing state information.
    :type statefile: str
    :var lastpkttime: Flag to control last packet time usage,
        if true, begin_time is appended to DATA command (Default is False).
    :type lastpkttime: boolean

    Protected parameters

    :var streams: Vector of SLNetStation objects.
    :type streams: list
    :var begin_time: Beginning of time window.
    :type begin_time: str
    :var end_time: End of time window.
    :type end_time: str
    :var resume: Flag to control resuming with sequence numbers.
    :type resume: boolean
    :var multistation: Flag to indicate multistation mode.
    :type multistation: boolean
    :var dialup: Flag to indicate dial-up mode.
    :type dialup: boolean
    :var terminate_flag: Flag to control connection termination.
    :type terminate_flag: boolean
    :var server_id: ID of the remote SeedLink server.
    :type server_id: str
    :var server_version: Version of the remote SeedLink server.
    :type server_version: float
    :var info_request_string: INFO level to request.
    :type info_request_string: str
    :var socket: The network socket.
    :type socket: :class:`socket.socket`
    :var state: Persistent state information.
    :type state: :class:`~obspy.seedlink.client.SLState`
    :var infoStrBuf: String to store INFO packet contents.
    :type infoStrBuf: str
    """

    SEEDLINK_PROTOCOL_PREFIX = "seedlink://"
    UNISTATION = "UNISTATION"
    UNINETWORK = "UNINETWORK"
    DFT_READBUF_SIZE = 1024
    QUOTE_CHAR = '"'

    def __init__(self, sllog=None):
        """
        Creates a new instance of SeedLinkConnection.

        :param sllog: SLLoc object to control info and error message logging.
        """
        self.sladdr = None
        self.keepalive = 0
        self.netto = 120
        self.netdly = 30
        self.sllog = None
        self.info_string = ""
        self.statefile = None
        self.lastpkttime = False
        self.streams = []
        self.begin_time = None
        self.end_time = None
        self.resume = True
        self.multistation = False
        self.dialup = False
        self.terminate_flag = False
        self.server_id = None
        self.server_version = 0.0
        self.info_request_string = None
        self.socket = None
        self.state = None
        self.infoStrBuf = ""

        self.state = SLState()
        if sllog is not None:
            self.sllog = sllog
        else:
            self.sllog = SLLog()

    def _log(self, is_error, verbosity, message):
        """
        Log helper method.
        """
        msg = "[%s] %s" % (self.sladdr, message)
        self.sllog.log(is_error, verbosity, msg)
        return msg

    def isConnected(self, timeout=1.0):
        """
        Returns connection state of the connection socket.

        :return: true if connected, false if not connected or socket is not
            initialized
        """
        return self.socket is not None and \
            self.isConnectedImpl(self.socket, timeout)

    def getState(self):
        """
        Returns the SLState state object.

        :return: the SLState state object
        """
        return self.state

    def setLog(self, sllog):
        """
        Sets the SLLog logging object.

        :param sllog: SLLoc object to control info and error message logging.
        """
        if sllog is not None:
            self.sllog = sllog

    def getLog(self):
        """
        Returns the SLLog logging object.

        :return: the SLLoc object to control info and error message logging.
        """
        return self.sllog

    def setNetTimout(self, netto):
        """
        Sets the network timeout (seconds).

        :param netto: the network timeout in seconds.
        """
        self.netto = netto

    def getNetTimout(self):
        """
        Returns the network timeout (seconds).

        :return: the network timeout in seconds.
        """
        return self.netto

    def setKeepAlive(self, keepalive):
        """
        Sets interval to send keepalive/heartbeat (seconds).

        :param keepalive: the interval to send keepalive/heartbeat in seconds.
        """
        self.keepalive = keepalive

    def getKeepAlive(self):
        """
        Returns the interval to send keepalive/heartbeat (seconds).

        :return: the interval to send keepalive/heartbeat in seconds.
        """
        return self.keepalive

    def setNetDelay(self, netdly):
        """
        Sets the network reconnect delay (seconds).

        :param netdly: the network reconnect delay in seconds.
        """
        self.netdly = netdly

    def getNetDelay(self):
        """
        Returns the network reconnect delay (seconds).

        :return: the network reconnect delay in seconds.
        """
        return self.netdly

    def setSLAddress(self, sladdr):
        """
        Sets the host:port of the SeedLink server.

        :param sladdr: the host:port of the SeedLink server.
        """
        prefix = SeedLinkConnection.SEEDLINK_PROTOCOL_PREFIX
        if sladdr.startswith(prefix):
            self.sladdr = len(sladdr[prefix:])
        self.sladdr = sladdr

    def setLastpkttime(self, lastpkttime):
        """
         Sets a specified start time for beginning of data transmission .

        :param lastpkttime: if true, beginning time of last packet received
            for each station is appended to DATA command on resume.
        """
        self.lastpkttime = lastpkttime

    def setBeginTime(self, startTimeStr):
        """
         Sets begin_time for initiation of continuous data transmission.

        :param startTimeStr: start time in in SeedLink string format:
            "year,month,day,hour,minute,second".
        """
        if startTimeStr is not None:
            self.begin_time = UTCDateTime(startTimeStr)
        else:
            self.begin_time = None

    def setEndTime(self, endTimeStr):
        """
         Sets end_time for termination of data transmission.

        :param endTimeStr: start time in in SeedLink string format:
            "year,month,day,hour,minute,second".
        """
        if endTimeStr is not None:
            self.end_time = UTCDateTime(endTimeStr)
        else:
            self.end_time = None

    def terminate(self):
        """"
        Sets terminate flag, closes connection and clears state.
        """
        self.terminate_flag = True

    def getSLAddress(self):
        """
        Returns the host:port of the SeedLink server.

        :return: the host:port of the SeedLink server.
        """
        return self.sladdr

    def getStreams(self):
        """
        Returns a copy of the Vector of SLNetStation objects.

        :return: a copy of the Vector of SLNetStation objects.
        """
        return list(self.streams)

    def getInfoString(self):
        """
        Returns the results of the last INFO request.

       :return: concatenation of contents of last terminated set of INFO
           packets
        """
        return self.info_string

    def createInfoString(self, strBuf):
        """
        Creates an info String from a String Buffer

        :param strBuf: the buffer to convert to an INFO String.

        :return: the INFO Sting.
        """
        strBuf = strBuf.replace("><", ">\n<")
        return str(strBuf).strip()

    def checkslcd(self):
        """
        Check this SeedLinkConnection description has valid parameters.

        :return: true if pass and false if problems were identified.
        """
        retval = True
        if len(self.streams) < 1 and self.info_request_string is None:
            self._log(True, 0, "stream chain AND info type are empty")
            retval = False
        ndx = 0
        if self.sladdr is None:
            self._log(False, 1, "server address %s is empty" % (self.sladdr))
            retval = False
        else:
            ndx = self.sladdr.find(':')
            if ndx < 1 or len(self.sladdr) < ndx + 2:
                self._log(True, 0, "host address " + \
                    "%s is not in '[hostname]:port' format" % (self.sladdr))
                retval = False
        return retval

    def readStreamList(self, streamfile, defselect):
        """
        Read a list of streams and selectors from a file and add them to the
        stream chain for configuring a multi-station connection.

        If 'defselect' is not null it will be used as the default selectors
        for entries will no specific selectors indicated.

        The file is expected to be repeating lines of the form:
        <PRE>
          <NET> <STA> [selectors]
        </PRE>
        For example:
        <PRE>
        # Comment lines begin with a '#' or '*'
        GE ISP  BH?.D
        NL HGN
        MN AQU  BH?  HH?
        </PRE>

        :param streamfile: name of file containing list of streams and
            selectors.
        :param defselect: default selectors.
        :return: the number of streams configured.

        :raise: SeedLinkException on error.
        """
        # Open the stream list file
        streamfile_file = None
        try:
            streamfile_file = open(streamfile, 'r')
        except IOError as ioe:
            self._log(True, 0, "cannot open state file %s" % (ioe))
            return 0
        except Exception as e:
            raise SeedLinkException(self._log(True, 0,
                "%s: opening state file: %s" % (e, streamfile)))
        self._log(False, 1,
            "recovering connection state from state file %s" % (streamfile))
        linecount = 0
        stacount = 0
        try:
            for line in streamfile_file:
                linecount += 1
                if line.startswith('#') or line.startswith('*'):
                    # comment lines
                    continue
                net = None
                station = None
                selectors_str = None
                tokens = line.split()
                if (len(tokens) >= 2):
                    net = tokens[0]
                    station = tokens[1]
                    selectors_str = ""
                    for token in tokens[2:]:
                        selectors_str += " " + token
                if net is None:
                    msg = "invalid or missing network string at line " + \
                        "%s of stream list file: %s"
                    self._log(True, 0, msg % (linecount, streamfile))
                    continue
                if station is None:
                    msg = "invalid or missing station string at line " + \
                        "%s of stream list file: %s"
                    self._log(True, 0, msg % (linecount, streamfile))
                    continue
                if selectors_str is not None:
                    self.addStream(net, station, selectors_str, -1, None)
                    stacount += 1
                else:
                    self.addStream(net, station, defselect, -1, None)
                    stacount += 1
            if (stacount == 0):
                self._log(True, 0, "no streams defined in %s" % (streamfile))
            else:
                self._log(False, 2,
                          "Read %s streams from %s" % (stacount, streamfile))
        except IOError as e:
            raise SeedLinkException(self._log(True, 0,
                "%s: reading stream list file: %s" % (e, streamfile)))
        finally:
            try:
                streamfile_file.close()
            except Exception as e:
                pass
        return stacount

    def parseStreamlist(self, streamlist, defselect):
        """
        Parse a string of streams and selectors and add them to the stream
        chain for configuring a multi-station connection.

        The string should be of the following form:
        "stream1[:selectors1],stream2[:selectors2],..."

        For example:
        <PRE>
        "IU_KONO:BHE BHN,GE_WLF,MN_AQU:HH?.D"
        </PRE>

        :param streamlist: list of streams and selectors.
        :param defselect: default selectors.

        :return: the number of streams configured.

        :raise: SeedLinkException on error.
        """
        # Parse the streams and selectors

        #print "DEBUG: streamlist:", streamlist
        stacount = 0
        for streamToken in streamlist.split(","):
            net = None
            station = None
            staselect = None
            configure = True
            try:
                reqTkz = streamToken.split(":")
                reqToken = reqTkz[0]
                netStaTkz = reqToken.split("_")
                # Fill in the NET and STA fields
                if (len(netStaTkz) != 2):
                    self._log(True, 0,
                        "not in NET_STA format: %s" % (reqToken))
                    configure = False
                else:
                    # First token, should be a network code
                    net = netStaTkz[0]
                    if len(net) < 1:
                        self._log(True, 0,
                            "not in NET_STA format: %s" % (reqToken))
                        configure = False
                    else:
                        # Second token, should be a station code
                        station = netStaTkz[1]
                        if len(station) < 1:
                            self._log(True, 0,
                                "not in NET_STA format: %s" % (reqToken))
                            configure = False
                    if len(reqTkz) > 1:
                        staselect = reqTkz[1]
                        if len(staselect) < 1:
                            self._log(True, 0,
                                "empty selector: %s" % (reqToken))
                            configure = False
                    else:
                        # If no specific selectors, use the default
                        staselect = defselect
                    #print "DEBUG: staselect:", staselect
                    # Add this to the stream chain
                    if configure:
                        try:
                            self.addStream(net, station, staselect, -1, None)
                            stacount += 1
                        except SeedLinkException as e:
                            raise e
            except Exception as e:
                raise e
        if (stacount == 0):
            self._log(True, 0, "no streams defined in stream list")
        else:
            if stacount > 0:
                self._log(False, 2,
                    "parsed %s streams from stream list" % (stacount))
        return stacount

    def addStream(self, net, station, selectors_str, seqnum, timestamp):
        """
        Add a new stream entry to the stream chain for the given net/station
        parameters.

        If the stream entry already exists do nothing and return 1.
        Also sets the multi-station flag to true.

        :param net: network code.
        :param station: station code.
        :param selectors: selectors for this net/station, null if none.
        :param seqnum: SeedLink sequence number of last packet received, -1 to
            start at the next data.
        :param timestamp: SeedLink time stamp in a UTCDateTime format
            for last packet received, null for none.

        :return: 0 if successfully added, 1 if an entry for network and station
            already exists.

        :raise: SeedLinkException on error.
        """
        # Sanity, check for a uni-station mode entry
        #print "DEBUG: selectors_str:", selectors_str
        if len(self.streams) > 0:
            stream = self.streams[0]
            if stream.net == SeedLinkConnection.UNINETWORK and \
               stream.station == SeedLinkConnection.UNISTATION:
                raise SeedLinkException(self.sllog.log(True, 0, "addStream" + \
                    "called, but uni-station mode already configured!"))
        selectors = selectors_str.split()

        # Search the stream chain if net/station/selector already present
        for stream in self.streams:
            if stream.net == net and stream.station == station:
                return stream.appendSelectors(selectors)

        # Add new stream
        newstream = SLNetStation(net, station, selectors, seqnum, timestamp)
        self.streams.append(newstream)
        self.multistation = True
        return 0

    def setUniParams(self, selectors_str, seqnum, timestamp):
        """
        Set the parameters for a uni-station mode connection for the
        given SLCD struct.  If the stream entry already exists, overwrite
        the previous settings.
        Also sets the multi-station flag to 0 (false).

        :param selectors: selectors for this net/station, null if none.
        :param seqnum: SeedLink sequence number of last packet received,
            -1 to start at the next data.
        :param timestamp: SeedLink time stamp in a UTCDateTime format
            for last packet received, null for none.

        :raise: SeedLinkException on error.
        """
        # Sanity, check for a multi-station mode entry
        if len(self.streams) > 0:
            stream = self.streams[0]
            if not stream.net == SeedLinkConnection.UNINETWORK or \
               not stream.station == SeedLinkConnection.UNISTATION:
                raise SeedLinkException(self._log(True, 0, "setUniParams " + \
                    "called, but multi-station mode already configured!"))
        selectors = None
        if selectors_str is not None and len(selectors_str) > 0:
            selectors = selectors_str.split()

        # Add new stream
        newstream = SLNetStation(SeedLinkConnection.UNINETWORK,
                                 SeedLinkConnection.UNISTATION, selectors,
                                 seqnum, timestamp)
        self.streams.append(newstream)
        self.multistation = False

    def setStateFile(self, statefile):
        """
        Set the state file and recover state.

        :param statefile: path and name of statefile.
        :return: the number of stream chains recovered.

        :raise: SeedLinkException on error.
        """
        self.statefile = statefile
        return self.recoverState(self.statefile)

    def recoverState(self, statefile):
        """
        Recover the state file and put the sequence numbers and time stamps
        into the pre-existing stream chain entries.

        :param statefile: path and name of statefile.
        :return: the number of stream chains recovered.

        :raise: SeedLinkException on error.
        """
        # open the state file
        statefile_file = None
        try:
            statefile_file = open(self.statefile, 'r')
        except IOError as ioe:
            self._log(True, 0, "cannot open state file: %s" % (ioe))
            return 0
        except Exception as e:
            raise SeedLinkException(self._log(True, 0,
                "%s: opening state file: %s" % (e, statefile)))

        # recover the state
        self._log(False, 1, "recovering connection state from state " + \
            "file: %s" % (self.statefile))
        linecount = 0
        stacount = 0
        try:
            for line in statefile_file:
                linecount += 1
                if line.startswith('#') or line.startswith('*'):
                    # comment lines
                    continue
                net = None
                station = None
                seqnum = -1
                timeStr = ""
                tokens = line.split()
                net = tokens[0]
                station = tokens[1]
                seqnum = int(tokens[2])
                timeStr = tokens[3]

                # check for completeness of read
                if timeStr == "":
                    self._log(True, 0,
                        "error parsing line of state file: %s" % (line))
                    continue
                else:
                    if timeStr == "null":
                        continue

                # Search for a matching net/station in the stream chain
                stream = None
                for i in range(len(self.streams)):
                    stream = self.streams[i]
                    if stream.net == net and stream.station == station:
                        break
                    stream = None

                # update net/station entry in the stream chain
                if stream is not None:
                    stream.seqnum = seqnum
                    if timeStr is not None:
                        try:
                            # AJL stream.btime = Btime(timeStr)
                            stream.btime = UTCDateTime(timeStr)
                            stacount += 1
                        except Exception as e:
                            self._log(True, 0, "parsing timestamp in line " + \
                                "%s of state file: %s" % (linecount, e.value))
            if (stacount == 0):
                self._log(True, 0,
                    "no matching streams found in %s" % (self.statefile))
            else:
                self._log(False, 2, "recoverd state for " + \
                    "%s streams in %s" % (stacount,  self.statefile))
        except IOError as e:
            raise SeedLinkException(self._log(True, 0,
                "%s: reading state file: %s" % (e, self.statefile)))
        finally:
            try:
                statefile_file.close()
            except Exception as e:
                pass
        return stacount

    def saveState(self, statefile):
        """
        Save all current s equence numbers and time stamps into the
        given state file.

        :param statefile: path and name of statefile.
        :return: the number of stream chains saved.

        :raise: SeedLinkException on error.
        """
        # open the state file
        statefile_file = None
        try:
            statefile_file = open(self.statefile, 'w')
        except IOError as ioe:
            self.sllog.log(True, 0, "[" + self.sladdr + "] cannot open state file: " + str(ioe))
            return 0
        except Exception as e:
            message = "[" + self.sladdr + "] " + e.value + ": opening state file: " + statefile
            self.sllog.log(True, 0, message)
            raise SeedLinkException(message)
        self.sllog.log(False, 2, "[" + self.sladdr + "] saving connection state to state file")
        stacount = 0
        try:
            # Loop through the stream chain
            for curstream in self.streams:
                #print "DEBUG: curstream:", curstream.net, curstream.station, curstream.btime
                if curstream.btime is not None:
                    statefile_file.write(curstream.net + " "
                                         + curstream.station + " "
                                         + str(curstream.seqnum) + " "
                                         + curstream.btime.formatSeedLink() + "\n")
        except IOError as e:
            message = "[" + self.sladdr + "] " + e.value + ": writing state file: " + self.statefile
            self.sllog.log(True, 0, message)
            raise SeedLinkException(message)
        finally:
            try:
                statefile_file.close()
            except Exception as e:
                pass
        return stacount

    def doTerminate(self):
        """
        Terminate the collection loop.
        """
        self.sllog.log(False, 0, "[" + self.sladdr + "] terminating collect loop")
        self.disconnect()
        self.state = SLState()
        self.info_request_string = None
        self.infoStrBuf = ""
        return SLPacket.SLTERMINATE

    def collect(self):
        """
        Manage a connection to a SeedLink server based on the values
        given in this SeedLinkConnection, and to collect data.

        Designed to run in a tight loop at the heart of a client program, this
        function will return every time a packet is received.

        :return: an SLPacket when something is received.
        :return: null when the connection was closed by
        the server or the termination sequence completed.

        :raise: SeedLinkException on error.
        """
        self.terminate_flag = False

        # Check if the infoRequestString was set
        if self.info_request_string is not None:
            self.state.query_mode = SLState.INFO_QUERY

        # If the connection is not up check this SeedLinkConnection and reset
        # the timing variables
        if self.socket is None or not self.isConnected():
            if not self.checkslcd():
                message = "[" + self.sladdr + "] problems with the connection description"
                self.sllog.log(True, 0, message)
                raise SeedLinkException(message)
            self.state.previous_time = time.time()
            #print "DEBUG: self.state.previous_time set:", self.state.previous_time
            self.state.netto_trig = -1
            self.state.keepalive_trig = -1

        # Start the primary loop
        npass = 0
        while True:

            self.sllog.log(False, 5, "[" + self.sladdr + "] primary loop pass " + str(npass))
            #print "DEBUG: self.state.state:", self.state.state
            npass += 1

            # we are terminating (abnormally!)
            if self.terminate_flag:
                return self.doTerminate()

            # not terminating
            if self.socket is None or not self.isConnected():
                self.state.state = SLState.SL_DOWN

            # Check for network timeout
            if (self.state.state == SLState.SL_DATA) and self.netto > 0 and self.state.netto_trig > 0:
                self.sllog.log(False, 0, "[" + self.sladdr + "] network timeout (" + str(self.netto) + "s), reconnecting in " + str(self.netdly) + "s")
                self.disconnect()
                self.state.state = SLState.SL_DOWN
                self.state.netto_trig = -1
                self.state.netdly_trig = -1

            # Check if a keepalive packet needs to be sent
            if (self.state.state == SLState.SL_DATA) and not self.state.expect_info and self.keepalive > 0 and self.state.keepalive_trig > 0:
                self.sllog.log(False, 2, "[" + self.sladdr + "] sending: keepalive request")
                try:
                    self.sendInfoRequest("ID", 3)
                    self.state.query_mode = SLState.KEEP_ALIVE_QUERY
                    self.state.expect_info = True
                    self.state.keepalive_trig = -1
                except IOError as ioe:
                    self.sllog.log(False, 0, "[" + self.sladdr + "] I/O error, reconnecting in " + str(self.netdly) + "s")
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN

            # Check if an in-stream INFO request needs to be sent
            if (self.state.state == SLState.SL_DATA) and not self.state.expect_info and self.info_request_string is not None:
                try:
                    self.sendInfoRequest(self.info_request_string, 1)
                    self.state.query_mode = SLState.INFO_QUERY
                    self.state.expect_info = True
                except IOError as ioe:
                    self.state.query_mode = SLState.NO_QUERY
                    self.sllog.log(False, 0, "[" + self.sladdr + "] I/O error, reconnecting in " + str(self.netdly) + "s")
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN
                self.info_request_string = None

            # Throttle the loop while delaying
            if (self.state.state == SLState.SL_DOWN) and self.state.netdly_trig > 0:
                time.sleep(0.5)

            # Connect to remote SeedLink server
            if (self.state.state == SLState.SL_DOWN) and (self.state.netdly_trig == 0):
                try:
                    self.connect()
                    self.state.state = SLState.SL_UP
                except Exception as e:
                    self.sllog.log(True, 0, e.value)
                    #traceback.print_exc()
                self.state.netto_trig = -1
                self.state.netdly_trig = -1

            # Negotiate/configure the connection
            if (self.state.state == SLState.SL_UP):

                # Send query if a query is set,
                #   stream configuration will be done only after query is fully returned
                if self.info_request_string is not None:
                    try:
                        self.sendInfoRequest(self.info_request_string, 1)
                        self.state.query_mode = SLState.INFO_QUERY
                        self.state.expect_info = True
                    except IOError as ioe:
                        self.sllog.log(False, 1, "[" + self.sladdr + "] SeedLink version does not support INFO requests")
                        self.state.query_mode = SLState.NO_QUERY
                        self.state.expect_info = False
                        self.sllog.log(True, 0, "[" + self.sladdr + "] I/O error, reconnecting in " + str(self.netdly) + "s")
                        self.disconnect()
                        self.state.state = SLState.SL_DOWN
                    self.info_request_string = None
                else:
                    if not self.state.expect_info:
                        try:
                            self.configLink()
                            self.state.recptr = 0
                            self.state.sendptr = 0
                            self.state.state = SLState.SL_DATA
                        except Exception as e:
                            self.sllog.log(True, 0, "[" + self.sladdr + "] negotiation with remote SeedLink failed: " + e.value)
                            self.disconnect()
                            self.state.state = SLState.SL_DOWN
                            self.state.netdly_trig = -1
                        self.state.expect_info = False

            # Process data in our buffer and then read incoming data
            if (self.state.state == SLState.SL_DATA) or self.state.expect_info and not (self.state.state == SLState.SL_DOWN):

                # Process data in buffer
                while self.state.packetAvailable():
                    slpacket = None
                    sendpacket = True

                    # Check for an INFO packet
                    if self.state.packetIsInfo():
                        terminator = chr(self.state.databuf[self.state.sendptr + SLPacket.SLHEADSIZE - 1]) != '*'
                        if not self.state.expect_info:
                            self.sllog.log(True, 0, "[" + self.sladdr + "] unexpected INFO packet received, skipping")
                        else:
                            if terminator:
                                self.state.expect_info = False

                            # Keep alive packets are not returned
                            if (self.state.query_mode == SLState.KEEP_ALIVE_QUERY):
                                sendpacket = False
                                if not terminator:
                                    self.sllog.log(True, 0, "[" + self.sladdr + "] non-terminated keep-alive packet received!?!")
                                else:
                                    self.sllog.log(False, 2, "[" + self.sladdr + "] keepalive packet received")
                            else:
                                slpacket = self.state.getPacket()
                                # construct info String
                                type = slpacket.getType()
                                #print "DEBUG: slpacket.getType():", slpacket.getType()
                                #print "DEBUG: SLPacket.TYPE_SLINF:", SLPacket.TYPE_SLINF
                                #print "DEBUG: SLPacket.TYPE_SLINFT:", SLPacket.TYPE_SLINFT
                                if (type == SLPacket.TYPE_SLINF):
                                    self.infoStrBuf += str(slpacket.msrecord)[64: len(slpacket.msrecord)]
                                else:
                                    if (type == SLPacket.TYPE_SLINFT):
                                        self.infoStrBuf += str(slpacket.msrecord)[64: len(slpacket.msrecord)]
                                        self.info_string = self.createInfoString(self.infoStrBuf)
                                        self.infoStrBuf = ""
                        if (self.state.query_mode != SLState.NO_QUERY):
                            self.state.query_mode = SLState.NO_QUERY
                    else:
                        # Get packet and update the stream chain entry if not an INFO packet
                        try:
                            slpacket = self.state.getPacket()
                            self.updateStream(slpacket)
                            if self.statefile is not None:
                                self.saveState(self.statefile)
                        except SeedLinkException as sle:
                            self.sllog.log(True, 0, "[" + self.sladdr + "] bad packet: " + sle.value)
                            sendpacket = False

                    # Increment the send pointer
                    self.state.incrementSendPointer()

                    # After processing the packet buffer shift the data
                    self.state.packDataBuffer()

                    # Return packet
                    if sendpacket:
                        return slpacket

                # A trap door for terminating, all complete data packets from the buffer
                #   have been sent to the caller
                # we are terminating (abnormally!)
                if self.terminate_flag:
                    return self.doTerminate()

                # Catch cases where the data stream stopped
                try:
                    if self.state.isError():
                        self.sllog.log(True, 0, "[" + self.sladdr + "] SeedLink reported an error with the last command")
                        self.disconnect()
                        return SLPacket.SLERROR
                except SeedLinkException as sle:
                    pass  # not enough bytes to determine packet type
                try:
                    if self.state.isEnd():
                        self.sllog.log(False, 1, "[" + self.sladdr + "] end of buffer or selected time window")
                        self.disconnect()
                        return SLPacket.SLTERMINATE
                except SeedLinkException as sle:
                    pass

                # Check for more available data from the socket
                bytesread = None
                try:
                    bytesread = self.receiveData(self.state.bytesRemaining(), self.sladdr)
                except IOError as ioe:
                    self.sllog.log(True, 0, "[" + self.sladdr + "] socket read error: " + str(ioe) + ", reconnecting in " + str(self.netdly) + "s")
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN
                    self.state.netto_trig = -1
                    self.state.netdly_trig = -1
                if bytesread is not None and len(bytesread) > 0:
                    # Data is here, process it
                    self.state.appendBytes(bytesread)
                    # Reset the timeout and keepalive timers
                    self.state.netto_trig = -1
                    self.state.keepalive_trig = -1
                else:
                    time.sleep(0.5)

            # Update timing variables when more than a 1/4 second has passed
            current_time = time.time()
            #print "DEBUG: if current_time - self.state.previous_time >= 0.25:", current_time, self.state.previous_time, current_time - self.state.previous_time
            if current_time - self.state.previous_time >= 0.25:
                #print "DEBUG: current_time - self.state.previous_time >= 0.25:", self.state.previous_time
                self.state.previous_time = time.time()
                #print "DEBUG: self.state.previous_time set:", self.state.previous_time

                # Network timeout timing logic
                if self.netto > 0:
                    if (self.state.netto_trig == -1):
                        self.state.netto_time = current_time
                        self.state.netto_trig = 0
                    else:
                        if (self.state.netto_trig == 0) and current_time - self.state.netto_time > self.netto:
                            self.state.netto_trig = 1
                #print "DEBUG: self.keepalive:", self.keepalive

                # Keepalive/heartbeat interval timing logic
                if self.keepalive > 0:
                    #print "DEBUG: self.state.keepalive_trig:", self.state.keepalive_trig
                    if (self.state.keepalive_trig == -1):
                        self.state.keepalive_time = current_time
                        self.state.keepalive_trig = 0
                    else:
                        #print "DEBUG: current_time - self.state.keepalive_time >=self.keepalive:", self.state.previous_time, current_time - self.state.keepalive_time, self.keepalive
                        if (self.state.keepalive_trig == 0) and current_time - self.state.keepalive_time > self.keepalive:
                            self.state.keepalive_trig = 1

                # Network delay timing logic
                if self.netdly > 0:
                    if (self.state.netdly_trig == -1):
                        self.state.netdly_time = current_time
                        self.state.netdly_trig = 1
                    else:
                        if (self.state.netdly_trig == 1) and current_time - self.state.netdly_time > self.netdly:
                            self.state.netdly_trig = 0

        # End of primary loop


    def connect(self):
        """
        Open a network socket connection to a SeedLink server.  Expects sladdr
        to be in 'host:port' format.

        :raise: SeedLinkException on error or no response or bad response from server.
        :raise: IOException if an I/O error occurs.
        """

        timeout = 10.0
        timeout = 4.0

        try:

            host_name = self.sladdr[0:self.sladdr.find(':')]
            nport = int(self.sladdr[self.sladdr.find(':') + 1:])

            # create and connect Socket
            sock = None
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            #print "DEBUG: sock.connect:", self.sladdr, host_name, nport
            sock.connect((host_name, nport));
            #print "DEBUG: sock.connect: sock:", sock
            if sock is None:
                raise Exception
            self.socket = sock

            # Check if socket is connected
            if not self.isConnected(timeout):
                message = "[" + self.sladdr + "] socket connect time-out (" + str(timeout) + "s)"
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
                raise SeedLinkException(message)

            # socket connected
            self.sllog.log(False, 1, "[" + self.sladdr + "] network socket opened")
            self.socket.settimeout(self.netto)

        except Exception as e:
            raise SeedLinkException("[" + self.sladdr + "] cannot connect to SeedLink server: " + e.value)

        # Everything should be connected, say hello
        try:
            self.sayHello()
        except SeedLinkException as sle:
            try:
                self.socket.close()
                self.socket = None
            except Exception:
                pass
            raise sle
        except IOError as ioe:
            #traceback.print_exc()
            try:
                self.socket.close()
                self.socket = None
            except Exception:
                pass
            raise ioe


    def disconnect(self):
        """
        Close the network socket associated with this connection.
        """
        if self.socket is not None:
            try:
                self.socket.close()
            except IOError as ioe:
                self.sllog.log(True, 1, "[" + self.sladdr + "] network socket close failed: " + str(ioe))
            self.socket = None
            self.sllog.log(False, 1, "[" + self.sladdr + "] network socket closed")

        # make sure previous state is cleaned up
        self.state = SLState()


    def close(self):
        """
        Closes this SeedLinkConnection by closing the network socket and saving the state to the statefile, if it exists.
        """
        if self.socket is not None:
            self.sllog.log(False, 1, "[" + self.sladdr + "] closing SeedLinkConnection()")
            self.disconnect()
        if self.statefile is not None:
            try:
                self.saveState(self.statefile)
            except SeedLinkException as sle:
                self.sllog.log(True, 0, sle.value)

    def isConnectedImpl(self, sock, timeout):
        """
        Check a socket for write ability using select()

        Time-out values are also passed (seconds) for the select() call.

        Returns:
         1 = success
         0 = if time-out expires
        -1 = errors

        """

        start_time = time.time()
        ready_to_read = []
        ready_to_write = []
        while (not sock in ready_to_write) and (time.time() - start_time) < timeout:
            ready_to_read, ready_to_write, in_error = \
            select.select([sock], [sock], [], timeout)

        #print "DEBUG: sock:", sock
        #print "DEBUG: ready_to_read:", ready_to_read
        #print "DEBUG: ready_to_write:", ready_to_write
        #print "DEBUG: in_error:", in_error
        if sock in ready_to_write:
            return True

        return False;


    def sendData(self, sendbytes, code, resplen):
        """
        Send bytes to the server. This is only designed for small pieces of data,
        specifically for when the server responses to commands.

        :param sendbytes: bytes to send.
        :param code: a string to include in error messages for identification.
        :param resplen: if > 0 then read up to resplen response bytes after sending.

        :return: the response bytes or null if no response requested.

        :raise: SeedLinkException on error or no response or bad response from server.
        :raise: IOException if an I/O error occurs.

        """
        #print "DEBUG: sendbytes:", repr(sendbytes)
        try:
            self.socket.send(sendbytes)
        except IOError as ioe:
            raise ioe

        if resplen <= 0:
            # no response requested
            return

        # If requested, wait up to 30 seconds for a response
        bytesread = None
        ackcnt = 0			# counter for the read loop
        ackpoll = 50                    # poll at 0.05 seconds for reading
        ackcntmax = 30000 / ackpoll     # 30 second wait
        bytesread = self.receiveData(resplen, code)
        while bytesread is not None and len(bytesread) == 0:
            if ackcnt > ackcntmax:
                raise SeedLinkException("[" + code + "] no response from SeedLink server to '" + str(sendbytes) + "'")
            sleep_time = 0.001 * ackpoll
            time.sleep(sleep_time)
            ackcnt += 1
            bytesread = self.receiveData(resplen, code)
        if bytesread is None:
            raise SeedLinkException("[" + code + "] bad response to '" + sendbytes + "'")
        return bytesread


    def receiveData(self, maxbytes, code):
        """
        Read bytes from the server.

        :param maxbytes: maximum number of bytes to read.
        :param code: a string to include in error messages for identification.

        :return: the response bytes (zero length if no available data), or null if EOF.

        :raise: IOException if an I/O error occurs.

        """


        bytesread = None

        # read up to maxbytes
        try:
            #self.socket.setblocking(0)
            bytesread = self.socket.recv(maxbytes)
            #self.socket.setblocking(1)
        except IOError as ioe:
            #traceback.print_exc()
            raise ioe
        #print "DEBUG: bytesread:", repr(bytesread)
        nbytesread = len(bytesread)

        # check for end or no bytes read
        if (nbytesread == -1):
            self.sllog.log(True, 1, "[" + code + "] socket.read(): " + str(nbytesread) + ": TCP FIN or EOF received")
            return
        else:
            if (nbytesread == 0):
                return ""

        return bytesread


    def sayHello(self):
        """
        Send the HELLO command and attempt to parse the server version
        number from the returned string.  The server version is set to 0.0
        if it can not be parsed from the returned string.

        :raise: SeedLinkException on error.
        :raise: IOException if an I/O error occurs.

        """

        sendStr = "HELLO"
        self.sllog.log(False, 2, "[" + self.sladdr + "] sending: " + sendStr)
        bytes = sendStr + "\r"
        bytesread = None
        bytesread = self.sendData(bytes, self.sladdr, SeedLinkConnection.DFT_READBUF_SIZE)

        # Parse the server ID and version from the returned string
        servstr = None
        try:
            servstr = str(bytesread)
            vndx = servstr.find(" v")
            if vndx < 0:
                self.server_id = servstr
                self.server_version = 0.0
            else:
                self.server_id = servstr[0:vndx]
                tmpstr = servstr[vndx + 2:]
                endndx = tmpstr.find(" ")
                #print "DEBUG: tmpstr:", tmpstr
                #print "DEBUG: tmpstr[0:endndx]:", tmpstr[0:endndx]
                self.server_version = float(tmpstr[0:endndx])
        except Exception as e:
            raise SeedLinkException("[" + self.sladdr + "] bad server ID/version string: '" + servstr + "'")

        # Check the response to HELLO
        if self.server_id.lower() == "SEEDLINK".lower():
            self.sllog.log(False, 1, "[" + self.sladdr + "] connected to: '" + servstr[0:servstr.find('\r')] + "'")
        else:
            raise SeedLinkException("[" + self.sladdr + "] incorrect response to HELLO: '" + servstr + "'")


    def requestInfo(self, infoLevel):
        """
        Add an INFO request to the SeedLink Connection Description.

        :param: infoLevel the INFO level (one of: ID, STATIONS, STREAMS, GAPS, CONNECTIONS, ALL)

        :raise: SeedLinkException if an INFO request is already pending.

        """
        if self.info_request_string is not None or self.state.expect_info:
            raise SeedLinkException("[" + self.sladdr + "] cannot make INFO request, one is already pending")
        else:
            self.info_request_string = infoLevel


    def sendInfoRequest(self, infoLevel, verb_level):
        """
        Sends a request for the specified INFO level.  The verbosity level
        can be specified, allowing control of when the request should be
        logged.

        :param: infoLevel the INFO level (one of: ID, STATIONS, STREAMS, GAPS, CONNECTIONS, ALL).

        :raise: SeedLinkException on error.
        :raise: IOException if an I/O error occurs.

        """

        if self.checkVersion(2.92) >= 0:
            bytes = "INFO " + infoLevel + "\r"
            self.sllog.log(False, verb_level, "[" + self.sladdr + "] sending: requesting INFO level " + infoLevel)
            self.sendData(bytes, self.sladdr, 0)
        else:
            raise SeedLinkException("[" + self.sladdr + "] detected SeedLink version (" + self.server_version + ") does not support INFO requests")


    def checkVersion(self, version):
        """
        Checks server version number against a given specified value.

        :param version: specified version value to test.

        :return:
         1 if version is greater than or equal to value specified,
         0 if no server version is known,
        -1 if version is less than value specified.

        """

        if (self.server_version == 0.0):
            return 0
        else:
            if self.server_version >= version:
                return 1
            else:
                return -1


    def configLink(self):
        """
        Configure/negotiate data stream(s) with the remote SeedLink
        server.  Negotiation will be either uni- or multi-station
        depending on the value of 'multistation' in this SeedLinkConnection.

        :raise: SeedLinkException on error.
        :raise: SeedLinkException if multi-station and SeedLink version does not support multi-station protocol.

        """
        if self.multistation:
            if self.checkVersion(2.5) >= 0:
                self.negotiateMultiStation()
            else:
                raise SeedLinkException("[" + self.sladdr + "] detected SeedLink version (" + self.server_version + ") does not support multi-station protocol")
        else:
            self.negotiateUniStation()

    def negotiateStation(self, curstream):
        """
        Negotiate a SeedLink connection for a single station and issue
        the DATA command.
        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :param: curstream the description of the station to negotiate.

        :raise: SeedLinkException on error.
        :raise: IOException if an I/O error occurs.

        """

        # Send the selector(s) and check the response(s)
        selectors = curstream.getSelectors()

        acceptsel = 0		# Count of accepted selectors
        for selector in selectors:
            if len(selector) > SLNetStation.MAX_SELECTOR_SIZE:
                self.sllog.log(False, 0, "[" + self.sladdr + "] invalid selector: " + selector)
            else:

                # Build SELECT command, send it and receive response
                sendStr = "SELECT " + selector
                self.sllog.log(False, 2, "[" + self.sladdr + "] sending: " + sendStr)
                bytes = sendStr + "\r"
                bytesread = None
                bytesread = self.sendData(bytes, self.sladdr, SeedLinkConnection.DFT_READBUF_SIZE)
                readStr = str(bytesread)

                # Check response to SELECT
                if readStr == "OK\r\n":
                    self.sllog.log(False, 2, "[" + self.sladdr + "] response: selector " + selector + " is OK")
                    acceptsel += 1
                else:
                    if readStr == "ERROR\r\n":
                        self.sllog.log(True, 0, "[" + self.sladdr + "] response: selector " + selector + " not accepted")
                    else:
                        raise SeedLinkException("[" + self.sladdr + "] response: invalid response to SELECT command: " + readStr)

        # Fail if none of the given selectors were accepted
        if acceptsel < 1:
            raise SeedLinkException("[" + self.sladdr + "] response: no data stream selector(s) accepted")
        else:
            self.sllog.log(False, 2, "[" + self.sladdr + "] response: " + str(acceptsel) + " selector(s) accepted")

        # Issue the DATA, FETCH or TIME action commands.  A specified start (and
        #   optionally, stop time) takes precedence over the resumption from any
        #   previous sequence number.

        sendStr = None
        if (curstream.seqnum != -1) and self.resume:
            if self.dialup:
                sendStr = "FETCH"
            else:
                sendStr = "DATA"

            # Append the last packet time if the feature is enabled and server is >= 2.93
            if self.lastpkttime and self.checkVersion(2.93) >= 0 and \
                    curstream.btime is not None:
                # Increment sequence number by 1
                sendStr += " " + hex(curstream.seqnum + 1) + " " + \
                    curstream.getSLTimeStamp()
                self.sllog.log(False, 1, 
                               "[" + self.sladdr + "] requesting resume data from 0x"
                               + hex(curstream.seqnum + 1).upper()
                               + " (decimal: " + str(curstream.seqnum + 1) + ") at "
                               + curstream.getSLTimeStamp())
            else:
                # Increment sequence number by 1
                sendStr += " " + hex(curstream.seqnum + 1)
                self.sllog.log(False, 1,
                               "[" + self.sladdr + "] requesting resume data from  0x"
                               + hex(curstream.seqnum + 1).upper()
                               + " (decimal: " + str(curstream.seqnum + 1) + ")")
        else:
            if self.begin_time is not None:
                # begin time specified (should only be at initial startup)
                if self.checkVersion(2.92) >= 0:
                    if self.end_time is None:
                        sendStr = "TIME " + self.begin_time.formatSeedLink()
                    else:
                        sendStr = "TIME " + self.begin_time.formatSeedLink() + " " + self.end_time.formatSeedLink()
                    self.sllog.log(False, 1, "[" + self.sladdr + "] requesting specified time window")
                else:
                    raise SeedLinkException("[" + self.sladdr + \
                        "] detected SeedLink version (" + \
                        self.server_version + \
                        ") does not support TIME windows")
            else:
                # default
                if self.dialup:
                    sendStr = "FETCH"
                else:
                    sendStr = "DATA"
                self.sllog.log(False, 1, "[" + self.sladdr + \
                               "] requesting next available data")

        # Send action command and receive response
        self.sllog.log(False, 2, "[" + self.sladdr + "] sending: " + sendStr)
        bytes = sendStr + "\r"
        bytesread = None
        bytesread = self.sendData(bytes, self.sladdr,
                                  SeedLinkConnection.DFT_READBUF_SIZE)

        # Check response to DATA/FETCH/TIME
        readStr = str(bytesread)
        if readStr == "OK\r\n":
            self.sllog.log(False, 2, "[" + self.sladdr + \
                           "] response: DATA/FETCH/TIME command is OK")
            acceptsel += 1
        else:
            if readStr == "ERROR\r\n":
                raise SeedLinkException("[" + self.sladdr + \
                    "] response: DATA/FETCH/TIME command is not accepted")
            else:
                raise SeedLinkException("[" + self.sladdr + \
                    "] response: invalid response to DATA/FETCH/TIME " + \
                    "command: " + readStr)

    def negotiateUniStation(self):
        """
        Negotiate a SeedLink connection in uni-station mode and issue the
        DATA command.  This is compatible with SeedLink Protocol version 2 or
        greater.

        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :raise: SeedLinkException on error.
        :raise: IOException if an I/O error occurs.
        """
        # get stream (should be only stream present)
        curstream = None
        try:
            curstream = self.streams[0]
        except Exception:
            raise SeedLinkException("[" + self.sladdr + \
                "] cannot negotiate uni-station, stream list does not " + \
                "have exactly one element")
        if not curstream.net == SeedLinkConnection.UNINETWORK and \
           curstream.station == SeedLinkConnection.UNISTATION:
            raise SeedLinkException("[" + self.sladdr + \
                "] cannot negotiate uni-station, mode not configured!")

        # negotiate the station connection
        self.negotiateStation(curstream)

    def negotiateMultiStation(self):
        """
        Negotiate a SeedLink connection using multi-station mode and
        issue the END action command.  This is compatible with SeedLink
        Protocol version 3, multi-station mode.
        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :raise: SeedLinkException on error.
        :raise: IOException if an I/O error occurs.
        """
        acceptsta = 0
        if len(self.streams) < 1:
            raise SeedLinkException("[" + self.sladdr + \
                "] cannot negotiate multi-station, stream list is empty")

        # Loop through the stream chain
        for curstream in self.streams:

            # A ring identifier
            #slring = curstream.net + curstream.station

            # Build STATION command, send it and receive response
            sendStr = "STATION  " + curstream.station + " " + curstream.net
            self.sllog.log(False, 2, "[" + self.sladdr + "] sending: " + \
                           sendStr)
            bytes = sendStr + "\r"
            bytesread = None
            bytesread = self.sendData(bytes, self.sladdr,
                                      SeedLinkConnection.DFT_READBUF_SIZE)
            readStr = str(bytesread)

            # Check response to SELECT
            if readStr == "OK\r\n":
                self.sllog.log(False, 2, "[" + self.sladdr + \
                               "] response: station is OK (selected)")
            else:
                if readStr == "ERROR\r\n":
                    self.sllog.log(True, 0, "[" + self.sladdr + \
                        "] response: station not accepted, skipping")
                    continue
                else:
                    raise SeedLinkException("[" + self.sladdr + \
                        "] response: invalid response to STATION command: " + \
                        readStr)

            # negotiate the station connection
            try:
                self.negotiateStation(curstream)
            except Exception as e:
                self.sllog.log(True, 0, e.value)
                continue
            acceptsta += 1

        # Fail if no stations were accepted
        if acceptsta < 1:
            raise SeedLinkException("[" + self.sladdr + \
                                    "] no stations accepted")
        else:
            self.sllog.log(False, 1, "[" + self.sladdr + "] " + \
                           str(acceptsta) + " station(s) accepted")

        # Issue END action command
        sendStr = "END"
        self.sllog.log(False, 2, "[" + self.sladdr + "] sending: " + sendStr)
        bytes = sendStr + "\r"
        self.sendData(bytes, self.sladdr, 0)

    def updateStream(self, slpacket):
        """
        Update the appropriate stream chain entry given a Mini-SEED record.

        :param: slpacket the packet conaining a Mini-SEED record.

        :raise: SeedLinkException on error.
        """
        seqnum = slpacket.getSequenceNumber()
        if (seqnum == -1):
            raise SeedLinkException("[" + self.sladdr + \
                                    "] could not determine sequence number")
        trace = None
        try:
            trace = slpacket.getTrace()
        except Exception as e:
            raise SeedLinkException("[" + self.sladdr + \
                "] blockette not 1000 (Data Only SEED Blockette) or other " + \
                "error reading miniseed data:" + e.value)

        # read some blockette fields
        net = None
        station = None
        btime = None
        try:
            station = trace.stats['station']
            net = trace.stats['network']
            btime = trace.stats['starttime']
            #print "DEBUG: station, net, btime:", station, net, btime
        except Exception as se:
            raise SeedLinkException("[" + self.sladdr + \
                                    "] trace header read error: " + se)

        # For uni-station mode
        if not self.multistation:
            curstream = None
            try:
                curstream = self.streams[0]
            except Exception as e:
                raise SeedLinkException("[" + self.sladdr + \
                    "] cannot update uni-station stream, stream list does " + \
                    "not have exactly one element")
            curstream.seqnum = seqnum
            curstream.btime = btime
            return

        # For multi-station mode, search the stream chain
        # Search for a matching net/station in the stream chain
        # AJL 20090306 - Add support for IRIS DMC enhancements:
        # Enhancements to the SeedLink protocol supported by the DMC's server
        # allow network and station codes to be
        # wildcarded in addition to the location and channel codes.
        wildcarded = False
        stream = None
        for stream in self.streams:
            if stream.net == net and stream.station == station:
                break
            if "?" in stream.net or "*" in stream.net or \
               "?" in stream.station or "*" in stream.station:
                # wildcard character found
                wildcarded = True
            stream = None
        #print "DEBUG: stream:", stream.net, stream.station, stream.btime

        # update net/station entry in the stream chain
        if stream is not None:
            stream.seqnum = seqnum
            stream.btime = btime
        else:
            if not wildcarded:
                self.sllog.log(True, 0, "unexpected data received: " + \
                               net + " " + station)
