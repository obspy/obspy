# -*- coding: utf-8 -*-
"""
Module to manage a connection to a SeedLink server using a Socket.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
import logging
import select
import socket
import time

from obspy.core.utcdatetime import UTCDateTime
from .slnetstation import SLNetStation
from .slstate import SLState
from ..seedlinkexception import SeedLinkException
from ..slpacket import SLPacket


# default logger
logger = logging.getLogger('obspy.clients.seedlink')


class SeedLinkConnection(object):
    """
    Class to manage a connection to a SeedLink server using a Socket.

    See obspy.realtime.seedlink.SLClient for an example of how to create
    and use this SeedLinkConnection object.
    A new SeedLink application can be created by sub-classing SLClient,
    or by creating a new class and invoking the methods of SeedLinkConnection.

    :var SEEDLINK_PROTOCOL_PREFIX: URI/URL prefix for seedlink
        servers ("seedlink://").
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
    :var info_string: String containing concatenation of contents of last
        terminated set of INFO packets.
    :type info_string: str
    :var statefile: File name for storing state information.
    :type statefile: str
    :var lastpkttime: Flag to control last packet time usage,
        if true, begin_time is appended to DATA command (Default is False).
    :type lastpkttime: bool
    :type timeout: float
    :param timeout: Time in seconds after which a `collect()` call will be
        interrupted.

    Protected parameters

    :var streams: Vector of SLNetStation objects.
    :type streams: list
    :var begin_time: Beginning of time window.
    :type begin_time: str
    :var end_time: End of time window.
    :type end_time: str
    :var resume: Flag to control resuming with sequence numbers.
    :type resume: bool
    :var multistation: Flag to indicate multistation mode.
    :type multistation: bool
    :var dialup: Flag to indicate dial-up mode.
    :type dialup: bool
    :var terminate_flag: Flag to control connection termination.
    :type terminate_flag: bool
    :var server_id: ID of the remote SeedLink server.
    :type server_id: str
    :var server_version: Version of the remote SeedLink server.
    :type server_version: float
    :var info_request_string: INFO level to request.
    :type info_request_string: str
    :var socket: The network socket.
    :type socket: :class:`socket.socket`
    :var state: Persistent state information.
    :type state: :class:`~obspy.clients.seedlink.client.SLState`
    :var infoStrBuf: String to store INFO packet contents.
    :type infoStrBuf: str
    """

    SEEDLINK_PROTOCOL_PREFIX = "seedlink://"
    SEEDLINK_DEFAULT_PORT = 18000
    UNISTATION = b"UNISTATION"
    UNINETWORK = b"UNINETWORK"
    DFT_READBUF_SIZE = 1024
    QUOTE_CHAR = b'"'

    def __init__(self, timeout=None):
        """
        Creates a new instance of SeedLinkConnection.
        """
        self.sladdr = None
        self.keepalive = 0
        self.netto = 120
        self.netdly = 30
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
        self.info_response_buffer = io.BytesIO()
        self.state = SLState()
        self.timeout = timeout

    def is_connected(self, timeout=1.0):
        """
        Returns connection state of the connection socket.

        :return: true if connected, false if not connected or socket is not
            initialized
        """
        return self.socket is not None and \
            self.is_connected_impl(self.socket, timeout)

    def get_state(self):
        """
        Returns the SLState state object.

        :return: the SLState state object
        """
        return self.state

    def set_net_timeout(self, netto):
        """
        Sets the network timeout (seconds).

        :param netto: the network timeout in seconds.
        """
        self.netto = netto

    def get_net_timeout(self):
        """
        Returns the network timeout (seconds).

        :return: the network timeout in seconds.
        """
        return self.netto

    def set_keep_alive(self, keepalive):
        """
        Sets interval to send keepalive/heartbeat (seconds).

        :param keepalive: the interval to send keepalive/heartbeat in seconds.
        """
        self.keepalive = keepalive

    def get_keep_alive(self):
        """
        Returns the interval to send keepalive/heartbeat (seconds).

        :return: the interval to send keepalive/heartbeat in seconds.
        """
        return self.keepalive

    def set_net_delay(self, netdly):
        """
        Sets the network reconnect delay (seconds).

        :param netdly: the network reconnect delay in seconds.
        """
        self.netdly = netdly

    def get_net_delay(self):
        """
        Returns the network reconnect delay (seconds).

        :return: the network reconnect delay in seconds.
        """
        return self.netdly

    def set_sl_address(self, sladdr):
        """
        Sets the host:port of the SeedLink server.

        :param sladdr: the host:port of the SeedLink server.
        """
        prefix = SeedLinkConnection.SEEDLINK_PROTOCOL_PREFIX
        if sladdr.startswith(prefix):
            sladdr = len(sladdr[prefix:])
        # use default port 18000
        if ':' not in sladdr:
            sladdr += ':%d' % self.SEEDLINK_DEFAULT_PORT
        self.sladdr = sladdr
        # set logger format
        name = " obspy.clients.seedlink [%s]" % (sladdr)
        logger.name = name

    def set_last_pkt_time(self, lastpkttime):
        """
         Sets a specified start time for beginning of data transmission .

        :param lastpkttime: if true, beginning time of last packet received
            for each station is appended to DATA command on resume.
        """
        self.lastpkttime = lastpkttime

    def set_begin_time(self, start_time_string):
        """
         Sets begin_time for initiation of continuous data transmission.

        :param start_time_string: start time in in SeedLink string format:
            "year,month,day,hour,minute,second".
        """
        if start_time_string is not None:
            self.begin_time = UTCDateTime(start_time_string)
        else:
            self.begin_time = None

    def set_end_time(self, end_time_string):
        """
         Sets end_time for termination of data transmission.

        :param end_time_string: start time in in SeedLink string format:
            "year,month,day,hour,minute,second".
        """
        if end_time_string is not None:
            self.end_time = UTCDateTime(end_time_string)
        else:
            self.end_time = None

    def terminate(self):
        """
        Sets terminate flag, closes connection and clears state.
        """
        self.terminate_flag = True

    def get_sl_address(self):
        """
        Returns the host:port of the SeedLink server.

        :return: the host:port of the SeedLink server.
        """
        return self.sladdr

    def get_streams(self):
        """
        Returns a copy of the Vector of SLNetStation objects.

        :return: a copy of the Vector of SLNetStation objects.
        """
        return list(self.streams)

    def get_info_string(self):
        """
        Returns the results of the last INFO request.

       :return: concatenation of contents of last terminated set of INFO
           packets
        """
        return self.info_string

    def check_slcd(self):
        """
        Check this SeedLinkConnection description has valid parameters.

        :return: true if pass and false if problems were identified.
        """
        retval = True
        if len(self.streams) < 1 and self.info_request_string is None:
            logger.error("stream chain AND info type are empty")
            retval = False
        ndx = 0
        if self.sladdr is None:
            logger.info("server address %s is empty" % (self.sladdr))
            retval = False
        else:
            ndx = self.sladdr.find(':')
            if ndx < 1 or len(self.sladdr) < ndx + 2:
                msg = "host address  %s is not in '[hostname]:port' format"
                logger.error(msg % (self.sladdr))
                retval = False
        return retval

    def read_stream_list(self, streamfile, defselect):
        """
        Read a list of streams and selectors from a file and add them to the
        stream chain for configuring a multi-station connection.

        If 'defselect' is not null it will be used as the default selectors
        for entries will no specific selectors indicated.

        The file is expected to be repeating lines of the form::

            <NET> <STA> [selectors]

        For example::

            # Comment lines begin with a '#' or '*'
            GE ISP  BH?.D
            NL HGN
            MN AQU  BH?  HH?

        :param streamfile: name of file containing list of streams and
            selectors.
        :param defselect: default selectors.
        :return: the number of streams configured.

        :raise SeedLinkException: on error.
        """
        # Open the stream list file
        streamfile_file = None
        try:
            streamfile_file = open(streamfile, 'r')
        except IOError as ioe:
            logger.error("cannot open state file %s" % (ioe))
            return 0
        except Exception as e:
            msg = "%s: opening state file: %s" % (e, streamfile)
            logger.critical(msg)
            raise SeedLinkException(msg)
        logger.info(
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
                    logger.error(msg % (linecount, streamfile))
                    continue
                if station is None:
                    msg = "invalid or missing station string at line " + \
                        "%s of stream list file: %s"
                    logger.error(msg % (linecount, streamfile))
                    continue
                if selectors_str is not None:
                    self.add_stream(net, station, selectors_str, -1, None)
                    stacount += 1
                else:
                    self.add_stream(net, station, defselect, -1, None)
                    stacount += 1
            if (stacount == 0):
                logger.error("no streams defined in %s" % (streamfile))
            else:
                logger.debug("Read %s streams from %s" % (stacount,
                                                          streamfile))
        except IOError as e:
            msg = "%s: reading stream list file: %s" % (e, streamfile)
            logger.critical(msg)
            raise SeedLinkException(msg)
        finally:
            try:
                streamfile_file.close()
            except Exception:
                pass
        return stacount

    def parse_stream_list(self, streamlist, defselect):
        """
        Parse a string of streams and selectors and add them to the stream
        chain for configuring a multi-station connection.

        The string should be of the following form::

            "stream1[:selectors1],stream2[:selectors2],..."

        For example::

            "IU_KONO:BHE BHN,GE_WLF,MN_AQU:HH?.D"

        :param streamlist: list of streams and selectors.
        :param defselect: default selectors.

        :return: the number of streams configured.

        :raise SeedLinkException: on error.
        """
        # Parse the streams and selectors

        # print("DEBUG: streamlist:", streamlist)
        stacount = 0
        for stream_token in streamlist.split(","):
            stream_token = stream_token.strip()
            net = None
            station = None
            staselect = None
            configure = True
            req_tkz = stream_token.split(":")
            req_token = req_tkz[0]
            net_sta_tkz = req_token.split("_")
            # Fill in the NET and STA fields
            if (len(net_sta_tkz) != 2):
                logger.error("not in NET_STA format: %s" % (req_token))
                configure = False
            else:
                # First token, should be a network code
                net = net_sta_tkz[0]
                if len(net) < 1:
                    logger.error("not in NET_STA format: %s" % (req_token))
                    configure = False
                else:
                    # Second token, should be a station code
                    station = net_sta_tkz[1]
                    if len(station) < 1:
                        logger.error("not in NET_STA format: %s" % (req_token))
                        configure = False
                if len(req_tkz) > 1:
                    staselect = req_tkz[1]
                    if len(staselect) < 1:
                        logger.error("empty selector: %s" % (req_token))
                        configure = False
                else:
                    # If no specific selectors, use the default
                    staselect = defselect
                # print("DEBUG: staselect:", staselect)
                # Add this to the stream chain
                if configure:
                    self.add_stream(net, station, staselect, -1, None)
                    stacount += 1
        if stacount == 0:
            logger.error("no streams defined in stream list")
        elif stacount > 0:
            msg = "parsed %d streams from stream list" % (stacount)
            logger.debug(msg)
        return stacount

    def add_stream(self, net, station, selectors_str, seqnum, timestamp):
        """
        Add a new stream entry to the stream chain for the given net/station
        parameters.

        If the stream entry already exists do nothing and return 1.
        Also sets the multi-station flag to true.

        :param net: network code.
        :param station: station code.
        :param selectors_str: selectors for this net/station, null if none.
        :param seqnum: SeedLink sequence number of last packet received, -1 to
            start at the next data.
        :param timestamp: SeedLink time stamp in a UTCDateTime format
            for last packet received, null for none.

        :return: 0 if successfully added, 1 if an entry for network and station
            already exists.

        :raise SeedLinkException: on error.
        """
        # Sanity, check for a uni-station mode entry
        # print("DEBUG: selectors_str:", selectors_str)
        if len(self.streams) > 0:
            stream = self.streams[0]
            if stream.net == SeedLinkConnection.UNINETWORK and \
               stream.station == SeedLinkConnection.UNISTATION:
                msg = "add_stream called, but uni-station mode configured!"
                logger.critical(msg)
                raise SeedLinkException(msg)

        if not selectors_str:
            selectors = []
        else:
            selectors = selectors_str.split()

        # Search the stream chain if net/station/selector already present
        for stream in self.streams:
            if stream.net == net and stream.station == station:
                return stream.append_selectors(selectors_str)

        # Add new stream
        newstream = SLNetStation(net, station, selectors, seqnum, timestamp)
        self.streams.append(newstream)
        self.multistation = True
        return 0

    def set_uni_params(self, selectors_str, seqnum, timestamp):
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

        :raise SeedLinkException: on error.
        """
        # Sanity, check for a multi-station mode entry
        if len(self.streams) > 0:
            stream = self.streams[0]
            if not stream.net == SeedLinkConnection.UNINETWORK or \
               not stream.station == SeedLinkConnection.UNISTATION:
                msg = "set_uni_params called, " \
                    "but multi-station mode configured!"
                logger.critical(msg)
                raise SeedLinkException(msg)
        selectors = None
        if selectors_str is not None and len(selectors_str) > 0:
            selectors = selectors_str.split()

        # Add new stream
        newstream = SLNetStation(SeedLinkConnection.UNINETWORK,
                                 SeedLinkConnection.UNISTATION, selectors,
                                 seqnum, timestamp)
        self.streams.append(newstream)
        self.multistation = False

    def set_state_file(self, statefile):
        """
        Set the state file and recover state.

        :param statefile: path and name of statefile.
        :return: the number of stream chains recovered.

        :raise SeedLinkException: on error.
        """
        self.statefile = statefile
        return self.recover_state(self.statefile)

    def recover_state(self, statefile):
        """
        Recover the state file and put the sequence numbers and time stamps
        into the pre-existing stream chain entries.

        :param statefile: path and name of statefile.
        :return: the number of stream chains recovered.

        :raise SeedLinkException: on error.
        """
        # open the state file
        statefile_file = None
        try:
            statefile_file = open(self.statefile, 'r')
        except IOError as ioe:
            logger.error("cannot open state file: %s" % (ioe))
            return 0
        except Exception as e:
            msg = "%s: opening state file: %s" % (e, statefile)
            logger.critical(msg)
            raise SeedLinkException(msg)

        # recover the state
        msg = "recovering connection state from state file: %s"
        logger.info(msg % (self.statefile))
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
                time_str = ""
                tokens = line.split()
                net = tokens[0]
                station = tokens[1]
                seqnum = int(tokens[2])
                time_str = tokens[3]

                # check for completeness of read
                if time_str == "":
                    msg = "error parsing line of state file: %s" % (line)
                    logger.error(msg)
                    continue
                elif time_str == "null":
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
                    if time_str is not None:
                        try:
                            # AJL stream.btime = Btime(timeStr)
                            stream.btime = UTCDateTime(time_str)
                            stacount += 1
                        except SeedLinkException as sle:
                            msg = "parsing timestamp in line %s of state " + \
                                "file: %s"
                            logger.error(msg % (linecount, sle.value))
                        except Exception as e:
                            msg = "parsing timestamp in line %s of state " + \
                                "file: %s"
                            logger.error(msg % (linecount, str(e)))
            if (stacount == 0):
                msg = "no matching streams found in %s"
                logger.error(msg % (self.statefile))
            else:
                msg = "recovered state for %s streams in %s"
                logger.debug(msg % (stacount, self.statefile))
        except IOError as e:
            msg = "%s: reading state file: %s" % (e, self.statefile)
            logger.critical(msg)
            raise SeedLinkException(msg)
        finally:
            try:
                statefile_file.close()
            except Exception:
                pass
        return stacount

    def save_state(self, statefile):
        """
        Save all current sequence numbers and time stamps into the
        given state file.

        :param statefile: path and name of statefile.
        :return: the number of stream chains saved.

        :raise SeedLinkException: on error.
        """
        # open the state file
        statefile_file = None
        try:
            statefile_file = open(self.statefile, 'w')
        except IOError as ioe:
            logger.error("cannot open state file: %s" % (ioe))
            return 0
        except Exception as e:
            msg = "%s: opening state file: %s" % (e, statefile)
            logger.critical(msg)
            raise SeedLinkException(msg)
        logger.debug("saving connection state to state file")
        stacount = 0
        try:
            # Loop through the stream chain
            for curstream in self.streams:
                # print("DEBUG: curstream:", curstream.net, curstream.station,
                #       curstream.btime)
                if curstream.btime is not None:
                    statefile_file.write(
                        curstream.net + " " +
                        curstream.station + " " + str(curstream.seqnum) +
                        " " + curstream.btime.format_seedlink() + "\n")
        except IOError as e:
            msg = "%s: writing state file: %s" % (e, self.statefile)
            logger.critical(msg)
            raise SeedLinkException(msg)
        finally:
            try:
                statefile_file.close()
            except Exception:
                pass
        return stacount

    def do_terminate(self):
        """
        Terminate the collection loop.
        """
        logger.warning("terminating collect loop")
        self.disconnect()
        self.state = SLState()
        self.info_request_string = None
        self.info_response_buffer = io.BytesIO()
        return SLPacket.SLTERMINATE

    def collect(self):
        """
        Manage a connection to a SeedLink server based on the values
        given in this SeedLinkConnection, and to collect data.

        Designed to run in a tight loop at the heart of a client program, this
        function will return every time a packet is received.

        If the SeedLinkConnection was initialized with a timeout, the collect()
        call will be terminated if it takes longer than `self.timeout` seconds
        to finish.

        :return: an SLPacket when something is received.
        :return: null when the connection was closed by
            the server or the termination sequence completed.

        :raise SeedLinkException: on error.
        """
        start_ = UTCDateTime()
        self.terminate_flag = False

        # Check if the infoRequestString was set
        if self.info_request_string is not None:
            self.state.query_mode = SLState.INFO_QUERY

        # If the connection is not up check this SeedLinkConnection and reset
        # the timing variables
        if self.socket is None or not self.is_connected():
            if not self.check_slcd():
                msg = "problems with the connection description"
                logger.critical(msg)
                raise SeedLinkException(msg)
            self.state.previous_time = time.time()
            # print("DEBUG: self.state.previous_time set:",
            #       self.state.previous_time)
            self.state.netto_trig = -1
            self.state.keepalive_trig = -1

        # Start the primary loop
        npass = 0
        while True:
            # manually check if we are over specified timeout
            if self.timeout is not None:
                if UTCDateTime() > start_ + self.timeout:
                    self.terminate_flag = True

            _msg = "primary loop pass %s, state %d"
            logger.debug(_msg % (npass, self.state.state))

            # we are terminating (abnormally!)
            if self.terminate_flag:
                return self.do_terminate()

            # not terminating
            if self.socket is None or not self.is_connected():
                self.state.state = SLState.SL_DOWN

            # Check for network timeout
            if self.state.state == SLState.SL_DATA and self.netto > 0 and \
               self.state.netto_trig > 0:
                msg = "network timeout (%s), reconnecting in %ss"
                logger.warn(msg % (self.netto, self.netdly))
                self.disconnect()
                self.state.state = SLState.SL_DOWN
                self.state.netto_trig = -1
                self.state.netdly_trig = -1

            # Check if a keepalive packet needs to be sent
            if self.state.state == SLState.SL_DATA and \
               not self.state.expect_info and self.keepalive > 0 and \
               self.state.keepalive_trig > 0:
                logger.debug("sending: keepalive request")
                try:
                    self.send_info_request("ID", 3)
                    self.state.query_mode = SLState.KEEP_ALIVE_QUERY
                    self.state.expect_info = True
                    self.state.keepalive_trig = -1
                except IOError:
                    msg = "I/O error, reconnecting in %ss"
                    logger.warn(msg % (self.netdly))
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN

            # Check if an in-stream INFO request needs to be sent
            if self.state.state == SLState.SL_DATA and \
               not self.state.expect_info and \
               self.info_request_string is not None:
                try:
                    self.send_info_request(self.info_request_string, 1)
                    self.state.query_mode = SLState.INFO_QUERY
                    self.state.expect_info = True
                except IOError:
                    self.state.query_mode = SLState.NO_QUERY
                    msg = "I/O error, reconnecting in %ss"
                    logger.warn(msg % (self.netdly))
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN
                self.info_request_string = None

            # Throttle the loop while delaying
            if self.state.state == SLState.SL_DOWN and \
               self.state.netdly_trig > 0:
                time.sleep(0.5)

            # Connect to remote SeedLink server
            if self.state.state == SLState.SL_DOWN and \
               self.state.netdly_trig == 0:
                self.connect()
                self.state.state = SLState.SL_UP
                self.state.netto_trig = -1
                self.state.netdly_trig = -1

            # Negotiate/configure the connection
            if self.state.state == SLState.SL_UP:

                # Send query if a query is set, stream configuration will be
                # done only after query is fully returned
                if self.info_request_string is not None:
                    try:
                        self.send_info_request(self.info_request_string, 1)
                        self.state.query_mode = SLState.INFO_QUERY
                        self.state.expect_info = True
                    except IOError:
                        msg = "SeedLink version does not support INFO requests"
                        logger.info(msg)
                        self.state.query_mode = SLState.NO_QUERY
                        self.state.expect_info = False
                        msg = "I/O error, reconnecting in %ss"
                        logger.warn(msg % (self.netdly))
                        self.disconnect()
                        self.state.state = SLState.SL_DOWN
                    self.info_request_string = None
                else:
                    if not self.state.expect_info:
                        try:
                            self.config_link()
                            self.state.recptr = 0
                            self.state.sendptr = 0
                            self.state.state = SLState.SL_DATA
                        except Exception as e:
                            msg = "negotiation with remote SeedLink failed: %s"
                            logger.error(msg % (e))
                            self.disconnect()
                            self.state.state = SLState.SL_DOWN
                            self.state.netdly_trig = -1
                        self.state.expect_info = False

            # Process data in our buffer and then read incoming data
            if self.state.state == SLState.SL_DATA or \
               self.state.expect_info and \
               not (self.state.state == SLState.SL_DOWN):

                # Process data in buffer
                while self.state.packet_available():
                    slpacket = None
                    sendpacket = True

                    # Check for an INFO packet
                    if self.state.packet_is_info():
                        temp = self.state.sendptr + SLPacket.SLHEADSIZE - 1
                        terminator = chr(self.state.databuf[temp]) != '*'
                        if not self.state.expect_info:
                            msg = "unexpected INFO packet received, skipping"
                            logger.error(msg)
                        else:
                            if terminator:
                                self.state.expect_info = False

                            # Keep alive packets are not returned
                            if self.state.query_mode == \
                               SLState.KEEP_ALIVE_QUERY:
                                sendpacket = False
                                if not terminator:
                                    logger.error(
                                        "non-terminated " +
                                        "keep-alive packet received!?!")
                                else:
                                    logger.debug("keepalive packet received")
                            else:
                                slpacket = self.state.get_packet()
                                # construct info String
                                packet_type = slpacket.get_type()
                                # print("DEBUG: slpacket.get_type():",
                                #       slpacket.get_type())
                                # print("DEBUG: SLPacket.TYPE_SLINF:",
                                #       SLPacket.TYPE_SLINF)
                                # print("DEBUG: SLPacket.TYPE_SLINFT:",
                                #       SLPacket.TYPE_SLINFT)
                                data = slpacket.get_string_payload()
                                self.info_response_buffer.write(data)

                                if (packet_type == SLPacket.TYPE_SLINFT):
                                    # Terminated INFO response packet
                                    # -> build complete INFO response string,
                                    #    strip NULL bytes from the end
                                    self.info_string = \
                                        self.info_response_buffer.getvalue().\
                                        decode('ASCII', errors='ignore').\
                                        replace("><", ">\n<").rstrip('\x00')

                                    self.info_response_buffer = io.BytesIO()
                        self.state.query_mode = SLState.NO_QUERY
                    else:
                        # Get packet and update the stream chain entry if not
                        # an INFO packet
                        try:
                            slpacket = self.state.get_packet()
                            self.update_stream(slpacket)
                            if self.statefile is not None:
                                self.save_state(self.statefile)
                        except SeedLinkException as sle:
                            logger.error("bad packet: %s" % (sle))
                            sendpacket = False

                    # Increment the send pointer
                    self.state.increment_send_pointer()

                    # After processing the packet buffer shift the data
                    self.state.pack_data_buffer()

                    # Return packet
                    if sendpacket:
                        return slpacket

                # A trap door for terminating, all complete data packets from
                # the buffer have been sent to the caller we are terminating
                # (abnormally!)
                if self.terminate_flag:
                    return self.do_terminate()

                # Catch cases where the data stream stopped
                try:
                    if self.state.is_error():
                        logger.error(
                            "SeedLink reported an error with the last command")
                        self.disconnect()
                        return SLPacket.SLERROR
                except SeedLinkException:
                    pass  # not enough bytes to determine packet type
                try:
                    if self.state.is_end():
                        logger.info("end of buffer or selected time window")
                        self.disconnect()
                        return SLPacket.SLTERMINATE
                except SeedLinkException:
                    pass

                # Check for more available data from the socket
                bytesread = None
                try:
                    bytesread = self.receive_data(self.state.bytes_remaining(),
                                                  self.sladdr)
                except IOError as ioe:
                    msg = "socket read error: %s, reconnecting in %sss"
                    logger.error(msg % (ioe, self.netdly))
                    self.disconnect()
                    self.state.state = SLState.SL_DOWN
                    self.state.netto_trig = -1
                    self.state.netdly_trig = -1
                if bytesread is not None and len(bytesread) > 0:
                    # Data is here, process it
                    self.state.append_bytes(bytesread)
                    # Reset the timeout and keepalive timers
                    self.state.netto_trig = -1
                    self.state.keepalive_trig = -1
                else:
                    time.sleep(0.5)

            # Update timing variables when more than a 1/4 second has passed
            now = time.time()
            # print("DEBUG: if now - self.state.previous_time >= 0.25:", now,
            #       self.state.previous_time, now - self.state.previous_time)
            if now - self.state.previous_time >= 0.25:
                # print("DEBUG: now - self.state.previous_time >= 0.25:",
                #       self.state.previous_time)
                self.state.previous_time = time.time()
                # print("DEBUG: self.state.previous_time set:",
                #       self.state.previous_time)

                # Network timeout timing logic
                if self.netto > 0:
                    if self.state.netto_trig == -1:
                        self.state.netto_time = now
                        self.state.netto_trig = 0
                    elif self.state.netto_trig == 0 and \
                            now - self.state.netto_time > self.netto:
                        self.state.netto_trig = 1
                # print("DEBUG: self.keepalive:", self.keepalive)

                # Keepalive/heartbeat interval timing logic
                if self.keepalive > 0:
                    # print("DEBUG: self.state.keepalive_trig:",
                    #       self.state.keepalive_trig)
                    # print("DEBUG: now - self.state.keepalive_time",
                    #       " >=self.keepalive:", self.state.previous_time,
                    #       now - self.state.keepalive_time, self.keepalive)
                    if self.state.keepalive_trig == -1:
                        self.state.keepalive_time = now
                        self.state.keepalive_trig = 0
                    elif self.state.keepalive_trig == 0 and \
                            now - self.state.keepalive_time > self.keepalive:
                        self.state.keepalive_trig = 1

                # Network delay timing logic
                if self.netdly > 0:
                    if self.state.netdly_trig == -1:
                        self.state.netdly_time = now
                        self.state.netdly_trig = 1
                    elif self.state.netdly_trig == 1 and \
                            now - self.state.netdly_time > self.netdly:
                        self.state.netdly_trig = 0
        # End of primary loop

    def connect(self):
        """
        Open a network socket connection to a SeedLink server. Expects sladdr
        to be in 'host:port' format.

        :raise SeedLinkException: on error or no response or bad response from
            server.
        :raise IOError: if an I/O error occurs.
        """
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
            # print("DEBUG: sock.connect:", self.sladdr, host_name, nport)
            sock.connect((host_name, nport))
            # print("DEBUG: sock.connect: sock:", sock)
            if sock is None:
                raise Exception
            self.socket = sock

            # Check if socket is connected
            if not self.is_connected(timeout):
                msg = "socket connect time-out %ss" % (timeout)
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
                raise SeedLinkException(msg)

            # socket connected
            logger.info("network socket opened")
            self.socket.settimeout(self.netto)

        except Exception as e:
            msg = "cannot connect to SeedLink server: %s"
            raise SeedLinkException(msg % (e))

        # Everything should be connected, say hello
        try:
            self.say_hello()
        except SeedLinkException as sle:
            try:
                self.socket.close()
                self.socket = None
            except Exception:
                pass
            raise sle
        except IOError as ioe:
            # traceback.print_exc()
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
                logger.error("network socket close failed: %s" % (ioe))
            self.socket = None
            logger.info("network socket closed")

        # make sure previous state is cleaned up
        self.state = SLState()

    def close(self):
        """
        Closes this SeedLinkConnection by closing the network socket and saving
        the state to the statefile, if it exists.
        """
        if self.socket is not None:
            logger.info("closing SeedLinkConnection()")
            self.disconnect()
        if self.statefile is not None:
            try:
                self.save_state(self.statefile)
            except SeedLinkException as sle:
                logger.error(sle.value)

    def is_connected_impl(self, sock, timeout):
        """
        Check a socket for write ability using select()

        Time-out values are also passed (seconds) for the select() call.

        :return: 1 = success, 0 = if time-out expires, -1 = errors
        """
        start_time = time.time()
        ready_to_write = []
        while (sock not in ready_to_write) and \
              (time.time() - start_time) < timeout:

            _ready_to_read, ready_to_write, _in_error = \
                select.select([sock], [sock], [], timeout)

        # print("DEBUG: sock:", sock)
        # print("DEBUG: ready_to_read:", ready_to_read)
        # print("DEBUG: ready_to_write:", ready_to_write)
        # print("DEBUG: in_error:", in_error)
        if sock in ready_to_write:
            return True
        return False

    def send_data(self, sendbytes, code, resplen):
        """
        Send bytes to the server. This is only designed for small pieces of
        data, specifically for when the server responses to commands.

        :param sendbytes: bytes to send.
        :param code: a string to include in error messages for identification.
        :param resplen: if > 0 then read up to resplen response bytes after
            sending.
        :return: the response bytes or null if no response requested.

        :raise SeedLinkException: on error or no or bad response from server.
        :raise IOError: if an I/O error occurs.

        """
        # print("DEBUG: sendbytes:", repr(sendbytes))
        try:
            self.socket.send(sendbytes)
        except IOError as ioe:
            raise ioe

        if resplen <= 0:
            # no response requested
            return

        # If requested, wait up to 30 seconds for a response
        ackcnt = 0  # counter for the read loop
        ackpoll = 50  # poll at 0.05 seconds for reading
        ackcntmax = 30000 / ackpoll  # 30 second wait
        bytesread = self.receive_data(resplen, code)
        while bytesread is not None and len(bytesread) == 0:
            if ackcnt > ackcntmax:
                msg = "[%s] no response from SeedLink server to '%s'"
                raise SeedLinkException(msg % (code, sendbytes))
            sleep_time = 0.001 * ackpoll
            time.sleep(sleep_time)
            ackcnt += 1
            bytesread = self.receive_data(resplen, code)
        if bytesread is None:
            msg = "[%s] bad response to '%s'"
            raise SeedLinkException(msg % (code, sendbytes))
        return bytesread

    def receive_data(self, maxbytes, code):
        """
        Read bytes from the server.

        :param maxbytes: maximum number of bytes to read.
        :param code: a string to include in error messages for identification.
        :return: the response bytes (zero length if no available data), or null
            if EOF.

        :raise IOError: if an I/O error occurs.
        """
        # read up to maxbytes
        try:
            # self.socket.setblocking(0)
            bytesread = self.socket.recv(maxbytes)
            # self.socket.setblocking(1)
        except IOError as ioe:
            # traceback.print_exc()
            raise ioe
        # print("DEBUG: bytesread:", repr(bytesread))
        nbytesread = len(bytesread)

        # check for end or no bytes read
        if (nbytesread == -1):
            # XXX This is never true
            msg = "[%s] socket.read(): %s: TCP FIN or EOF received"
            logger.error(msg % (code, nbytesread))
            return
        else:
            if (nbytesread == 0):
                return b""

        return bytesread

    def say_hello(self):
        """
        Send the HELLO command and attempt to parse the server version
        number from the returned string.  The server version is set to 0.0
        if it can not be parsed from the returned string.

        :raise SeedLinkException: on error.
        :raise IOError: if an I/O error occurs.
        """
        send_str = b"HELLO"
        logger.debug("sending: %s" % (send_str.decode()))
        bytes_ = send_str + b"\r"
        bytesread = self.send_data(bytes_, self.sladdr,
                                   SeedLinkConnection.DFT_READBUF_SIZE)

        # Parse the server ID and version from the returned string
        servstr = None
        try:
            servstr = bytesread.decode()
            vndx = servstr.find(" v")
            if vndx < 0:
                self.server_id = servstr
                self.server_version = 0.0
            else:
                self.server_id = servstr[0:vndx]
                tmpstr = servstr[vndx + 2:]
                endndx = tmpstr.find(" ")
                # print("DEBUG: tmpstr:", tmpstr)
                # print("DEBUG: tmpstr[0:endndx]:", tmpstr[0:endndx])
                self.server_version = float(tmpstr[0:endndx])
        except Exception:
            msg = "bad server ID/version string: '%s'"
            raise SeedLinkException(msg % (servstr))

        # Check the response to HELLO
        if self.server_id.lower() == "seedlink":
            msg = "connected to: '" + servstr[0:servstr.find('\r')] + "'"
            logger.info(msg)
        else:
            msg = "ncorrect response to HELLO: '%s'" % (servstr)
            raise SeedLinkException(msg)

    def request_info(self, info_level):
        """
        Add an INFO request to the SeedLink Connection Description.

        :param info_level: the INFO level (one of: ID, STATIONS, STREAMS, GAPS,
            CONNECTIONS, ALL)

        :raise SeedLinkException: if an INFO request is already pending.
        """
        if self.info_request_string is not None or self.state.expect_info:
            msg = "cannot make INFO request, one is already pending"
            raise SeedLinkException(msg)
        else:
            self.info_request_string = info_level

    def send_info_request(self, info_level, verb_level):
        """
        Sends a request for the specified INFO level. The verbosity level
        can be specified, allowing control of when the request should be
        logged.

        :param info_level: the INFO level (one of: ID, STATIONS, STREAMS, GAPS,
            CONNECTIONS, ALL).

        :raise SeedLinkException: on error.
        :raise IOError: if an I/O error occurs.
        """
        if self.check_version(2.92) >= 0:
            bytes_ = b"INFO " + info_level.encode('ascii', 'strict') + b"\r"
            msg = "sending: requesting INFO level %s" % (info_level)
            if verb_level == 1:
                logger.info(msg)
            else:
                logger.debug(msg)
            self.send_data(bytes_, self.sladdr, 0)
        else:
            msg = "detected SeedLink version %s does not support INFO requests"
            raise SeedLinkException(msg % (self.server_version))

    def check_version(self, version):
        """
        Checks server version number against a given specified value.

        :param version: specified version value to test.
        :return: 1 if version is greater than or equal to value specified,
             0 if no server version is known, -1 if version is less than value
             specified.
        """
        if (self.server_version == 0.0):
            return 0
        else:
            if self.server_version >= version:
                return 1
            else:
                return -1

    def config_link(self):
        """
        Configure/negotiate data stream(s) with the remote SeedLink
        server.  Negotiation will be either uni- or multi-station
        depending on the value of 'multistation' in this SeedLinkConnection.

        :raise SeedLinkException: on error.
        :raise SeedLinkException: if multi-station and SeedLink version does
            not support multi-station protocol.
        """
        if self.multistation:
            if self.check_version(2.5) >= 0:
                self.negotiate_multi_station()
            else:
                msg = "detected SeedLink version %s does not support " + \
                    "multi-station protocol"
                raise SeedLinkException(msg % (self.server_version))
        else:
            self.negotiate_uni_station()

    def negotiate_station(self, curstream):
        """
        Negotiate a SeedLink connection for a single station and issue
        the DATA command.
        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :param curstream: the description of the station to negotiate.

        :raise SeedLinkException: on error.
        :raise IOError: if an I/O error occurs.

        """

        # Send the selector(s) and check the response(s)
        selectors = curstream.get_selectors()

        acceptsel = 0  # Count of accepted selectors
        for selector_str in selectors:
            selector = selector_str.encode('ascii', 'strict')
            if len(selector) > SLNetStation.MAX_SELECTOR_SIZE:
                logger.warn("invalid selector: %s" % (selector))
            else:
                # Build SELECT command, send it and receive response
                send_str = b"SELECT " + selector
                logger.debug("sending: %s" % (send_str))
                bytes_ = send_str + b"\r"
                bytesread = None
                bytesread = self.send_data(bytes_, self.sladdr,
                                           SeedLinkConnection.DFT_READBUF_SIZE)
                read_str = bytesread.decode()

                # Check response to SELECT
                if read_str == "OK\r\n":
                    logger.debug("response: selector %s is OK" % (selector))
                    acceptsel += 1
                elif read_str == "ERROR\r\n":
                    msg = "response: selector %s not accepted"
                    logger.error(msg % (selector))
                else:
                    msg = "response: invalid response to SELECT command: %s"
                    raise SeedLinkException(msg % (read_str))

        # Fail if none of the given selectors were accepted
        if acceptsel < 1:
            msg = "response: no data stream selector(s) accepted"
            raise SeedLinkException(msg)

        msg = "response: %s selector(s) accepted"
        logger.debug(msg % (acceptsel))

        # Issue the DATA, FETCH or TIME action commands. A specified start (and
        # optionally, stop time) takes precedence over the resumption from any
        # previous sequence number.

        send_str = None
        if (curstream.seqnum != -1) and self.resume:
            if self.dialup:
                send_str = b"FETCH"
            else:
                send_str = b"DATA"

            # Append the last packet time if the feature is enabled and server
            # is >= 2.93
            if self.lastpkttime and self.check_version(2.93) >= 0 and \
               curstream.btime is not None:
                # Increment sequence number by 1
                send_str += b" " + \
                    hex(curstream.seqnum + 1).encode('ascii', 'strict') + \
                    b" " + curstream.get_sl_time_stamp()
                msg = "requesting resume data from 0x%s (decimal: %s) at %s"
                logger.info(msg % (hex(curstream.seqnum + 1).upper(),
                            curstream.seqnum + 1),
                            curstream.get_sl_time_stamp())
            else:
                # Increment sequence number by 1
                send_str += b" " + \
                    hex(curstream.seqnum + 1).encode('ascii', 'strict')
                msg = "requesting resume data from 0x%s (decimal: %s)"
                logger.info(msg % (hex(curstream.seqnum + 1).upper(),
                                   curstream.seqnum + 1))
        elif self.begin_time is not None:
            # begin time specified (should only be at initial startup)
            if self.check_version(2.92) >= 0:
                send_str = b"TIME " + self.begin_time.\
                    format_seedlink().encode('ascii', 'strict')
                if self.end_time is not None:
                    send_str += b" " + self.end_time.format_seedlink().\
                        encode('ascii', 'strict')
                logger.info("requesting specified time window")
            else:
                msg = "detected SeedLink version %s does not support " + \
                    "TIME windows"
                raise SeedLinkException(msg % (self.server_version))
        else:
            # default
            if self.dialup:
                send_str = b"FETCH"
            else:
                send_str = b"DATA"
            logger.info("requesting next available data")

        # Send action command and receive response
        logger.debug("sending: %s" % (send_str))
        bytes_ = send_str + b"\r"
        bytesread = None
        bytesread = self.send_data(bytes_, self.sladdr,
                                   SeedLinkConnection.DFT_READBUF_SIZE)

        # Check response to DATA/FETCH/TIME
        read_str = bytesread.decode()
        if read_str == "OK\r\n":
            logger.debug("response: DATA/FETCH/TIME command is OK")
            acceptsel += 1
        elif read_str == "ERROR\r\n":
            msg = "response: DATA/FETCH/TIME command is not accepted"
            raise SeedLinkException(msg)
        else:
            msg = "response: invalid response to DATA/FETCH/TIME command: %s"
            raise SeedLinkException(msg % (read_str))

    def negotiate_uni_station(self):
        """
        Negotiate a SeedLink connection in uni-station mode and issue the
        DATA command.  This is compatible with SeedLink Protocol version 2 or
        greater.

        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :raise SeedLinkException: on error.
        :raise IOError: if an I/O error occurs.
        """
        # get stream (should be only stream present)
        curstream = None
        try:
            curstream = self.streams[0]
        except Exception:
            msg = "cannot negotiate uni-station, stream list does not " + \
                "have exactly one element"
            raise SeedLinkException(msg)
        if not curstream.net == SeedLinkConnection.UNINETWORK and \
           curstream.station == SeedLinkConnection.UNISTATION:
            msg = "cannot negotiate uni-station, mode not configured!"
            raise SeedLinkException(msg)
        # negotiate the station connection
        self.negotiate_station(curstream)

    def negotiate_multi_station(self):
        """
        Negotiate a SeedLink connection using multi-station mode and
        issue the END action command.  This is compatible with SeedLink
        Protocol version 3, multi-station mode.
        If selectors are defined, then the string is parsed on space and each
        selector is sent.
        If 'seqnum' != -1 and the SLCD 'resume' flag is true then data is
        requested starting at seqnum.

        :raise SeedLinkException: on error.
        :raise IOError: if an I/O error occurs.
        """
        acceptsta = 0
        if len(self.streams) < 1:
            msg = "cannot negotiate multi-station, stream list is empty"
            raise SeedLinkException(msg)

        # Loop through the stream chain
        for curstream in self.streams:

            # A ring identifier
            # slring = curstream.net + curstream.station

            # Build STATION command, send it and receive response
            send_str = ("STATION  " + curstream.station + " " +
                        curstream.net).encode('ascii', 'strict')
            logger.debug("sending: %s" % send_str.decode())
            bytes_ = send_str + b"\r"
            bytesread = self.send_data(bytes_, self.sladdr,
                                       SeedLinkConnection.DFT_READBUF_SIZE)
            read_str = bytesread

            # Check response to SELECT
            if read_str == b"OK\r\n":
                logger.debug("response: station is OK (selected)")
            elif read_str == b"ERROR\r\n":
                logger.error("response: station not accepted, skipping")
                continue
            else:
                msg = "response: invalid response to STATION command: %s"
                raise SeedLinkException(msg % (read_str))

            # negotiate the station connection
            try:
                self.negotiate_station(curstream)
            except SeedLinkException as sle:
                logger.error(sle.value)
                continue
            except Exception as e:
                logger.error(str(e))
                continue
            acceptsta += 1

        # Fail if no stations were accepted
        if acceptsta < 1:
            raise SeedLinkException("no stations accepted")

        logger.info("%s station(s) accepted" % (acceptsta))

        # Issue END action command
        send_str = b"END"
        logger.debug("sending: %s" % (send_str.decode()))
        bytes_ = send_str + b"\r"
        self.send_data(bytes_, self.sladdr, 0)

    def update_stream(self, slpacket):
        """
        Update the appropriate stream chain entry given a Mini-SEED record.

        :param slpacket: the packet containing a Mini-SEED record.

        :raise SeedLinkException: on error.
        """
        seqnum = slpacket.get_sequence_number()
        if (seqnum == -1):
            raise SeedLinkException("could not determine sequence number")
        trace = None
        try:
            trace = slpacket.get_trace()
        except Exception as e:
            msg = "blockette not 1000 (Data Only SEED Blockette) or other " + \
                "error reading miniseed data: %s"
            raise SeedLinkException(msg % (e))

        # read some blockette fields
        net = None
        station = None
        btime = None
        try:
            station = trace.stats['station']
            net = trace.stats['network']
            btime = trace.stats['starttime']
            # print("DEBUG: station, net, btime:", station, net, btime)
        except Exception as se:
            raise SeedLinkException("trace header read error: %s" % (se))

        # For uni-station mode
        if not self.multistation:
            curstream = None
            try:
                curstream = self.streams[0]
            except Exception:
                msg = "cannot update uni-station stream, stream list does " + \
                    "not have exactly one element"
                raise SeedLinkException(msg)
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
        # print("DEBUG: stream:", stream.net, stream.station, stream.btime)

        # update net/station entry in the stream chain
        if stream is not None:
            stream.seqnum = seqnum
            stream.btime = btime
        elif not wildcarded:
            logger.error("unexpected data received: %s %s" % (net, station))
