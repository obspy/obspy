# -*- coding: utf-8 -*-
import numpy as np

from obspy.core.utcdatetime import UTCDateTime
from obspy.realtime.rttrace import RtTrace
from obspy.clients.seedlink.slclient import SLClient
from obspy.clients.seedlink.slpacket import SLPacket


class MySLClient(SLClient):
    """
    A custom SeedLink client.
    """
    def __init__(self, rt_trace=RtTrace(), *args, **kwargs):
        """
        Creates a new instance of SLClient accepting a realtime trace handler.
        """
        self.rt_trace = rt_trace
        super(self.__class__, self).__init__(*args, **kwargs)

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
            print("-" * 40)
            print("Complete INFO:\n" + self.slconn.get_info_string())
            if self.infolevel is not None:
                return True
            else:
                return False

        # can send an in-line INFO request here
        if (count % 100 == 0):
            infostr = "ID"
            self.slconn.request_info(infostr)

        # if here, must be a data blockette
        print("-" * 40)
        print(self.__class__.__name__ + ": packet seqnum:", end=' ')
        print(str(seqnum) + ": blockette type: " + str(type))

        # process packet data
        trace = slpack.get_trace()
        if trace is not None:
            print(self.__class__.__name__ +
                  ": blockette contains a trace: ", end=' ')
            print(trace.id, trace.stats['starttime'], end=' ')
            print(" dt:" + str(1.0 / trace.stats['sampling_rate']), end=' ')
            print(" npts:" + str(trace.stats['npts']), end=' ')
            print(" sampletype:" + str(trace.stats['sampletype']), end=' ')
            print(" dataquality:" + str(trace.stats['dataquality']))
            # Custom: append packet data to RtTrace
            # g_o_check = True    # raises Error on gap or overlap
            g_o_check = False  # clears RTTrace memory on gap or overlap
            self.rt_trace.append(trace, gap_overlap_check=g_o_check,
                                 verbose=True)
            length = self.rt_trace.stats.npts / \
                self.rt_trace.stats.sampling_rate
            print(self.__class__.__name__ + ":", end=' ')
            print("append to RTTrace: npts:",
                  str(self.rt_trace.stats.npts), end=' ')
            print("length:" + str(length) + "s")
            # post processing to do something interesting
            peak = np.amax(np.abs(self.rt_trace.data))
            print(self.__class__.__name__ + ": abs peak = " + str(peak))
        else:
            print(self.__class__.__name__ + ": blockette contains no trace")
        return False


def main():
    # initialize realtime trace
    rttrace = RtTrace(max_length=60)
    # rttrace.register_rt_process('integrate')
    rttrace.register_rt_process(np.abs)
    # width in num samples
    boxcar_width = 10 * int(rttrace.stats.sampling_rate + 0.5)
    rttrace.register_rt_process('boxcar', width=boxcar_width)

    print("The SeedLink client will collect data packets and append " +
          "them to an RTTrace object.")

    # create SeedLink client
    sl_client = MySLClient(rt_trace=rttrace)
    #
    sl_client.slconn.set_sl_address("geofon.gfz-potsdam.de:18000")
    sl_client.multiselect = ("GE_STU:BHZ")
    #
    # slClient.slconn.set_sl_address("discovery.rm.ingv.it:39962")
    # slClient.multiselect = ("IV_MGAB:BHZ")
    #
    # slClient.slconn.set_sl_address("rtserve.iris.washington.edu:18000")
    # slClient.multiselect = ("AT_TTA:BHZ")
    #
    # set a time window from 2 min in the past to 5 sec in the future
    dt = UTCDateTime()
    sl_client.begin_time = (dt - 120.0).format_seedlink()
    sl_client.end_time = (dt + 5.0).format_seedlink()
    print("SeedLink date-time range:", sl_client.begin_time, " -> ",
          end=' ')
    print(sl_client.end_time)
    sl_client.verbose = 3
    sl_client.initialize()
    sl_client.run()


if __name__ == '__main__':
    main()
