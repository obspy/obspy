#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to create and use a connection to a SeedLink server using a
SeedLinkConnection object.

A new SeedLink application can be created by sub-classing SLClient and
overriding at least the packetHandler method of SLClient.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.seedlink.client.seedlinkconnection import SeedLinkConnection
from obspy.seedlink.seedlinkexception import SeedLinkException
from obspy.seedlink.sllog import SLLog
from obspy.seedlink.slpacket import SLPacket
import sys
import traceback


class SLClient(object):
    """
    Basic class to create and use a connection to a SeedLink server using a
    SeedLinkConnection object.

    A new SeedLink application can be created by sub-classing SLClient and
    overriding at least the packetHandler method of SLClient.

    :var slconn: SeedLinkConnection object for communicating with the
        SeedLinkConnection over a socket.
    :type slconn: SeedLinkConnection
    :var verbose: Verbosity level, 0 is lowest.
    :type verbose: int
    :var ppackets: Flag to indicate show detailed packet information.
    :type  ppackets: boolean
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
    :type  begin_time :str
    :var end_time: End of time window for reading windowed data.
    :type  end_time: str
    :var infolevel: INFO LEVEL for info request only.
    :type  infolevel: str
    :var sllog: Logging object.
    :type  sllog SLLog
    """
    VERSION = "1.2.0X00"
    VERSION_YEAR = "2011"
    VERSION_DATE = "24Nov" + VERSION_YEAR
    COPYRIGHT_YEAR = VERSION_YEAR
    PROGRAM_NAME = "SLClient v" + VERSION
    VERSION_INFO = PROGRAM_NAME + " (" + VERSION_DATE + ")"
    BANNER = ["SLClient comes with ABSOLUTELY NO WARRANTY"]

    def __init__(self, sllog=None):
        """
        Creates a new instance of SLClient with the specified logging object

        :param: sllog logging object to handle messages.
        """
        self.slconn = None
        self.verbose = 0
        self.ppackets = False
        self.streamfile = None
        self.selectors = None
        self.multiselect = None
        self.statefile = None
        self.begin_time = None
        self.end_time = None
        self.infolevel = None
        self.sllog = None

        ## for-while
        for line in SLClient.BANNER:
            print line
        self.sllog = sllog
        self.slconn = SeedLinkConnection(self.sllog)

    def parseCmdLineArgs(self, args):
        """
        Parses the commmand line arguments.

        :type args: list
        :param args: main method arguments.
        :return: -1 on error, 1 if version or help argument found, 0 otherwise.
        """
        if len(args) < 2:
            self.printUsage(False)
            return 1
        optind = 1
        while optind < len(args):
            if args[optind] == "-V":
                print(self.VERSION_INFO, sys.stderr)
                return 1
            elif args[optind] == "-h":
                self.printUsage(False)
                return 1
            elif args[optind].startswith("-v"):
                self.verbose += len(args[optind]) - 1
            elif args[optind] == "-p":
                self.ppackets = True
            elif args[optind] == "-nt":
                optind += 1
                self.slconn.setNetTimout(int(args[optind]))
            elif args[optind] == "-nd":
                optind += 1
                self.slconn.setNetDelay(int(args[optind]))
            elif args[optind] == "-k":
                optind += 1
                self.slconn.setKeepAlive(int(args[optind]))
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
                print("Unknown option: " + args[optind], sys.stderr)
                return -1
            elif self.slconn.getSLAddress() is None:
                self.slconn.setSLAddress(args[optind])
            else:
                print("Unknown option: " + args[optind], sys.stderr)
                return -1
            optind += 1
        return 0

    def initialize(self):
        """
        Initializes this SLClient.
        """
        if self.slconn.getSLAddress() is None:
            message = "no SeedLink server specified"
            raise SeedLinkException(message)

        if self.sllog is None:
            self.sllog = SLLog(self.verbose, None, None, None, None)
        self.slconn.setLog(self.sllog)
        if self.verbose >= 2:
            self.ppackets = True
# XXX: java class InetAddress?
#        if self.slconn.getSLAddress().startswith(":"):
#            str(self.slconn.setSLAddress(InetAddress.getLocalHost()) + \
#                self.slconn.getSLAddress())
        if self.streamfile is not None:
            self.slconn.readStreamList(self.streamfile, self.selectors)
        if self.multiselect is not None:
            self.slconn.parseStreamlist(self.multiselect, self.selectors)
        else:
            if self.streamfile is None:
                self.slconn.setUniParams(self.selectors, -1, None)
        if self.statefile is not None:
            self.slconn.setStateFile(self.statefile)
        else:
            if self.begin_time is not None:
                self.slconn.setBeginTime(self.begin_time)
            if self.end_time is not None:
                self.slconn.setEndTime(self.end_time)

    def run(self):
        """
        Start this SLClient.
        """
        if self.infolevel is not None:
            self.slconn.requestInfo(self.infolevel)
        # Loop with the connection manager
        count = 1
        slpack = self.slconn.collect()
        while slpack is not None:
            if (slpack == SLPacket.SLTERMINATE):
                break
            try:
                # do something with packet
                terminate = self.packetHandler(count, slpack)
                if terminate:
                    break
            except SeedLinkException as sle:
                print(self.__class__.__name__ + ": " + sle.value)
            if count >= sys.maxint:
                count = 1
                print "DEBUG INFO: " + self.__class__.__name__ + ":",
                print "Packet count reset to 1"
            else:
                count += 1
            slpack = self.slconn.collect()

        # Close the SeedLinkConnection
        self.slconn.close()

    def packetHandler(self, count, slpack):
        """
        Processes each packet received from the SeedLinkConnection.

        This mehod should be overridded when subclassing SLClient.

        :type count: int
        :param count:  Packet counter.
        :type trace: :class:`~obspy.seedlink.SLPacket`
        :param trace:  :class:`~obspy.seedlink.SLPacket` the packet to process.
        :return: Boolean true if connection to SeedLink server should be
            closed and session terminated, false otherwise.
        """
        # check if not a complete packet
        if slpack is None or (slpack == SLPacket.SLNOPACKET) or \
                (slpack == SLPacket.SLERROR):
            return False

        # get basic packet info
        seqnum = slpack.getSequenceNumber()
        type = slpack.getType()

        # process INFO packets here
        if (type == SLPacket.TYPE_SLINF):
            return False
        if (type == SLPacket.TYPE_SLINFT):
            print "Complete INFO:\n" + self.slconn.getInfoString()
            if self.infolevel is not None:
                return True
            else:
                return False

        # can send an in-line INFO request here
        try:
            #if (count % 100 == 0 and not self.slconn.state.expect_info):
            if (count % 100 == 0):
                infostr = "ID"
                self.slconn.requestInfo(infostr)
        except SeedLinkException as sle:
            print(self.__class__.__name__ + ": " + sle.value)

        # if here, must be a data blockette
        print self.__class__.__name__ + ": packet seqnum:",
        print str(seqnum) + ": blockette type: " + str(type)
        if not self.ppackets:
            return False

        # process packet data
        trace = slpack.getTrace()
        if trace is not None:
            print self.__class__.__name__ + ": blockette contains a trace: ", \
            trace.stats['network'], trace.stats['station'], \
            trace.stats['location'], trace.stats['channel'], \
            trace.stats['starttime'], \
            " dt:" + str(trace.stats['sampling_rate']), \
            " npts:" + str(trace.stats['npts']), \
            " sampletype:" + str(trace.stats['sampletype']), \
            " dataquality:" + str(trace.stats['dataquality'])
            if self.verbose >= 3:
                print self.__class__.__name__ + ":"
                print "blockette contains a trace: " + str(trace.stats)
        else:
            print self.__class__.__name__ + ": blockette contains no trace"
        return False

    def printUsage(self, concise):
        """
        Prints the usage message for this class.
        """
        print("\nUsage: python %s [options] <[host]:port>" % \
              (self.__class__.__name__))
        if concise:
            print("Use '-h' for detailed help")
            return
        print
        print(" ## General program options ##")
        print(" -V             report program version")
        print(" -h             show this usage message")
        print(" -v             be more verbose, multiple flags can be used")
        print(" -p             print details of data packets")
        print(" -nd delay      network re-connect delay (seconds), default 30")
        print(" -nt timeout    network timeout (seconds), re-establish connection if no")
        print("                  data/keepalives are received in this time, default 600")
        print(" -k interval    send keepalive (heartbeat) packets this often (seconds)")
        print(" -x statefile   save/restore stream state information to this file")
        print(" -t begintime   sets a beginning time for the initiation of data transmission (year,month,day,hour,minute,second)")
        print(" -e endtime     sets an end time for windowed data transmission  (year,month,day,hour,minute,second)")
        print(" -i infolevel   request this INFO level, write response to std out, and exit ")
        print("                  infolevel is one of: ID, STATIONS, STREAMS, GAPS, CONNECTIONS, ALL ")
        print
        print(" ## Data stream selection ##")
        print(" -l listfile    read a stream list from this file for multi-station mode")
        print(" -s selectors   selectors for uni-station or default for multi-station")
        print(" -S streams     select streams for multi-station (requires SeedLink >= 2.5)")
        print("   'streams' = 'stream1[:selectors1],stream2[:selectors2],...'")
        print("        'stream' is in NET_STA format, for example:")
        print("        -S \"IU_KONO:BHE BHN,GE_WLF,MN_AQU:HH?.D\"")
        print("")
        print(" <[host]:port>  Address of the SeedLink server in host:port format")
        print("                  if host is omitted (i.e. ':18000'), localhost is assumed")

    @classmethod
    def main(cls, args):
        """
        Main method - creates and runs an SLClient using the specified
        command line arguments
        """
        slClient = None
        try:
            slClient = SLClient()
            rval = slClient.parseCmdLineArgs(args)
            if (rval != 0):
                sys.exit(rval)
            slClient.initialize()
            slClient.run()
        except Exception as e:
            if slClient is not None and slClient.sllog is not None:
                slClient.sllog.log(True, 0, e)
            else:
                sys.stderr.write("Error: " + e.value)
            traceback.print_exc()


if __name__ == '__main__':
    SLClient.main(sys.argv)
