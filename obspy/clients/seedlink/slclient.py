#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to create and use a connection to a SeedLink server using a
SeedLinkConnection object.

A new SeedLink application can be created by sub-classing SLClient and
overriding at least the packet_handler method of SLClient.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import logging
import sys
import traceback

from .client.seedlinkconnection import SeedLinkConnection
from .seedlinkexception import SeedLinkException
from .slpacket import SLPacket


USAGE = """
## General program options ##
-V             report program version
-h             show this usage message
-v             be more verbose, multiple flags can be used
-p             print details of data packets
-nd delay      network re-connect delay (seconds), default 30
-nt timeout    network timeout (seconds), re-establish connection if no
               data/keepalives are received in this time, default 600
-k interval    send keepalive (heartbeat) packets this often (seconds)
-x statefile   save/restore stream state information to this file
-t begintime   sets a beginning time for the initiation of data transmission
               (year,month,day,hour,minute,second)
-e endtime     sets an end time for windowed data transmission
               (year,month,day,hour,minute,second)
-i infolevel   request this INFO level, write response to std out, and exit
               infolevel is one of: ID, STATIONS, STREAMS, GAPS, CONNECTIONS,
               ALL

## Data stream selection ##
-l listfile    read a stream list from this file for multi-station mode
-s selectors   selectors for uni-station or default for multi-station
-S streams     select streams for multi-station (requires SeedLink >= 2.5)
  'streams' = 'stream1[:selectors1],stream2[:selectors2],...'
       'stream' is in NET_STA format, for example:
       -S \"IU_KONO:BHE BHN,GE_WLF,MN_AQU:HH?.D\"

<[host]:port>  Address of the SeedLink server in host:port format
               if host is omitted (i.e. ':18000'), localhost is assumed
"""


# default logger
logger = logging.getLogger('obspy.clients.seedlink')


class SLClient(object):
    """
    Basic class to create and use a connection to a SeedLink server using a
    SeedLinkConnection object.

    A new SeedLink application can be created by sub-classing SLClient and
    overriding at least the packet_handler method of SLClient.

    :var slconn: SeedLinkConnection object for communicating with the
        SeedLinkConnection over a socket.
    :type slconn: SeedLinkConnection
    :var verbose: Verbosity level, 0 is lowest.
    :type verbose: int
    :var ppackets: Flag to indicate show detailed packet information.
    :type  ppackets: bool
    :var streamfile: Name of file containing stream list for multi-station
        mode.
    :type  streamfile: str
    :var selectors: Selectors for uni-station or default selectors for
        multi-station.
    :type  selectors: str
    :var multiselect: Selectors for multi-station.
    :type  multiselect: str
    :var statefile: Name of file for reading (if exists) and storing state.
    :type  statefile: str
    :var begin_time: Beginning of time window for read start in past.
    :type  begin_time: str
    :var end_time: End of time window for reading windowed data.
    :type  end_time: str
    :var infolevel: INFO LEVEL for info request only.
    :type  infolevel: str
    :type timeout: float
    :param timeout: Timeout in seconds, passed on to the underlying
        SeedLinkConnection.
    """
    VERSION = "1.2.0X00"
    VERSION_YEAR = "2011"
    VERSION_DATE = "24Nov" + VERSION_YEAR
    COPYRIGHT_YEAR = VERSION_YEAR
    PROGRAM_NAME = "SLClient v" + VERSION
    VERSION_INFO = PROGRAM_NAME + " (" + VERSION_DATE + ")"

    def __init__(self, loglevel='DEBUG', timeout=None):
        """
        Creates a new instance of SLClient with the specified logging object
        """
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)

        self.verbose = 0
        self.ppackets = False
        self.streamfile = None
        self.selectors = None
        self.multiselect = None
        self.statefile = None
        self.begin_time = None
        self.end_time = None
        self.infolevel = None
        self.timeout = timeout
        self.slconn = SeedLinkConnection(timeout=timeout)

    def parse_cmd_line_args(self, args):
        """
        Parses the command line arguments.

        :type args: list
        :param args: main method arguments.
        :return: -1 on error, 1 if version or help argument found, 0 otherwise.
        """
        if len(args) < 2:
            self.print_usage(False)
            return 1
        optind = 1
        while optind < len(args):
            if args[optind] == "-V":
                print(self.VERSION_INFO, file=sys.stderr)
                return 1
            elif args[optind] == "-h":
                self.print_usage(False)
                return 1
            elif args[optind].startswith("-v"):
                self.verbose += len(args[optind]) - 1
            elif args[optind] == "-p":
                self.ppackets = True
            elif args[optind] == "-nt":
                optind += 1
                self.slconn.set_net_timeout(int(args[optind]))
            elif args[optind] == "-nd":
                optind += 1
                self.slconn.set_net_delay(int(args[optind]))
            elif args[optind] == "-k":
                optind += 1
                self.slconn.set_keep_alive(int(args[optind]))
            elif args[optind] == "-l":
                optind += 1
                self.streamfile = args[optind]
            elif args[optind] == "-s":
                optind += 1
                self.selectors = args[optind]
            elif args[optind] == "-S":
                optind += 1
                self.multiselect = args[optind]
            elif args[optind] == "-x":
                optind += 1
                self.statefile = args[optind]
            elif args[optind] == "-t":
                optind += 1
                self.begin_time = args[optind]
            elif args[optind] == "-e":
                optind += 1
                self.end_time = args[optind]
            elif args[optind] == "-i":
                optind += 1
                self.infolevel = args[optind]
            elif args[optind].startswith("-"):
                print("Unknown option: " + args[optind], file=sys.stderr)
                return -1
            elif self.slconn.get_sl_address() is None:
                self.slconn.set_sl_address(args[optind])
            else:
                print("Unknown option: " + args[optind], file=sys.stderr)
                return -1
            optind += 1
        return 0

    def initialize(self):
        """
        Initializes this SLClient.
        """
        if self.slconn.get_sl_address() is None:
            message = "no SeedLink server specified"
            raise SeedLinkException(message)

        if self.verbose >= 2:
            self.ppackets = True
        if self.slconn.get_sl_address().startswith(":"):
            self.slconn.set_sl_address("127.0.0.1" +
                                       self.slconn.get_sl_address())
        if self.streamfile is not None:
            self.slconn.read_stream_list(self.streamfile, self.selectors)
        if self.multiselect is not None:
            self.slconn.parse_stream_list(self.multiselect, self.selectors)
        else:
            if self.streamfile is None:
                self.slconn.set_uni_params(self.selectors, -1, None)
        if self.statefile is not None:
            self.slconn.set_state_file(self.statefile)
        else:
            if self.begin_time is not None:
                self.slconn.set_begin_time(self.begin_time)
            if self.end_time is not None:
                self.slconn.set_end_time(self.end_time)

    def run(self, packet_handler=None):
        """
        Start this SLClient.

        :type packet_handler: func
        :param packet_handler: Custom packet handler funtion to override
            `self.packet_handler` for this seedlink request. The function will
            be repeatedly called with two arguments: the current packet counter
            (`int`) and the currently served seedlink packet
            (:class:`~obspy.clients.seedlink.SLPacket`). The function should
            return `True` to abort the request or `False` to continue the
            request.
        """
        if packet_handler is None:
            packet_handler = self.packet_handler
        if self.infolevel is not None:
            self.slconn.request_info(self.infolevel)
        # Loop with the connection manager
        count = 1
        slpack = self.slconn.collect()
        while slpack is not None:
            if (slpack == SLPacket.SLTERMINATE):
                break
            try:
                # do something with packet
                terminate = packet_handler(count, slpack)
                if terminate:
                    break
            except SeedLinkException as sle:
                print(self.__class__.__name__ + ": " + sle.value)
            if count >= sys.maxsize:
                count = 1
                print("DEBUG INFO: " + self.__class__.__name__ + ":", end=' ')
                print("Packet count reset to 1")
            else:
                count += 1
            slpack = self.slconn.collect()

        # Close the SeedLinkConnection
        self.slconn.close()

    def packet_handler(self, count, slpack):
        """
        Processes each packet received from the SeedLinkConnection.

        This method should be overridden when sub-classing SLClient.

        :type count: int
        :param count:  Packet counter.
        :type slpack: :class:`~obspy.clients.seedlink.slpacket.SLPacket`
        :param slpack: packet to process.

        :rtype: bool
        :return: True if connection to SeedLink server should be closed and
            session terminated, False otherwise.
        """
        # check if not a complete packet
        if slpack is None or (slpack == SLPacket.SLNOPACKET) or \
                (slpack == SLPacket.SLERROR):
            return False

        # get basic packet info
        seqnum = slpack.get_sequence_number()
        type = slpack.get_type()

        # process INFO packets here
        if (type == SLPacket.TYPE_SLINF):
            return False
        if (type == SLPacket.TYPE_SLINFT):
            print("Complete INFO:\n" + self.slconn.get_info_string())
            if self.infolevel is not None:
                return True
            else:
                return False

        # can send an in-line INFO request here
        try:
            # if (count % 100 == 0 and not self.slconn.state.expect_info):
            if (count % 100 == 0):
                infostr = "ID"
                self.slconn.request_info(infostr)
        except SeedLinkException as sle:
            print(self.__class__.__name__ + ": " + sle.value)

        # if here, must be a data blockette
        print(self.__class__.__name__ + ": packet seqnum:", end=' ')
        print(str(seqnum) + ": blockette type: " + str(type))
        if not self.ppackets:
            return False

        # process packet data
        trace = slpack.get_trace()
        if trace is not None:
            print(self.__class__.__name__ + ": blockette contains a trace: ")
            print(trace.id, trace.stats['starttime'], end=' ')
            print(" dt:" + str(1.0 / trace.stats['sampling_rate']), end=' ')
            print(" npts:" + str(trace.stats['npts']), end=' ')
            print(" sampletype:" + str(trace.stats['sampletype']), end=' ')
            print(" dataquality:" + str(trace.stats['dataquality']))
            if self.verbose >= 3:
                print(self.__class__.__name__ + ":")
                print("blockette contains a trace: " + str(trace.stats))
        else:
            print(self.__class__.__name__ + ": blockette contains no trace")
        return False

    def print_usage(self, concise=True):
        """
        Prints the usage message for this class.
        """
        print("\nUsage: python %s [options] <[host]:port>" %
              (self.__class__.__name__))
        if concise:
            usage = "Use '-h' for detailed help"
        else:
            usage = USAGE
        print(usage)

    @classmethod
    def main(cls, args):
        """
        Main method - creates and runs an SLClient using the specified
        command line arguments
        """
        try:
            sl_client = SLClient()
            rval = sl_client.parse_cmd_line_args(args)
            if (rval != 0):
                sys.exit(rval)
            sl_client.initialize()
            sl_client.run()
        except Exception as e:
            logger.critical(e)
            traceback.print_exc()


if __name__ == '__main__':
    SLClient.main(sys.argv)
