# -*- coding: utf-8 -*-
"""
Module to manage SeedLinkConnection state.

Part of Python implementation of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.seedlink.seedlinkexception import SeedLinkException
from obspy.seedlink.slpacket import SLPacket


class SLState(object):
    """
    Class to manage SeedLinkConnection state.

    :var SL_DOWN: Connection state down.
    :type SL_DOWN: int
    :var SL_UP: Connection state up.
    :type SL_UP: int
    :var SL_DATA: Connection state data.
    :type SL_DATA: int
    :var state: Connection state.
    :type state: int
    :var NO_QUERY: INFO query state NO_QUERY.
    :type NO_QUERY: int
    :var INFO_QUERY: INFO query state INFO_QUERY.
    :type INFO_QUERY: int
    :var KEEP_ALIVE_QUERY: INFO query state KEEP_ALIVE_QUERY.
    :type KEEP_ALIVE_QUERY: int
    :var query_mode: INFO query state.
    :type query_mode: int
    :var BUFSIZE: Size of receiving buffer (default is 8192).
    :type BUFSIZE: int
    :var databuf: Data buffer for received packets.
    :type databuf: bytearray
    :var recptr: Receive pointer for databuf.
    :type recptr: int
    :var sendptr: Send pointer for databuf.
    :type sendptr: int
    :var expect_info: Flag to indicate if an INFO response is expected.
    :type expect_info: bool
    :var netto_trig: Network timeout trigger.netto_trig
    :type netto_trig: int
    :var netdly_trig: Network re-connect delay trigger.
    :type netdly_trig: int
    :var keepalive_trig: Send keepalive trigger.
    :type keepalive_trig: int
    :var previous_time: Time stamp of last state update.
    :type previous_time: float
    :var netto_time: Network timeout time stamp.
    :type netto_time: float
    :var netdly_time: Network re-connect delay time stamp.
    :type netdly_time: float
    :var keepalive_time: Keepalive time stamp.
    :type keepalive_time: float
    """
    SL_DOWN = 0
    SL_UP = 1
    SL_DATA = 2
    NO_QUERY = 0
    INFO_QUERY = 1
    KEEP_ALIVE_QUERY = 2
    BUFSIZE = 8192

    def __init__(self):
        self.state = SLState.SL_DOWN
        self.query_mode = SLState.NO_QUERY
        # AJL self.databuf = [str() for __idx0 in range(BUFSIZE)]
        self.databuf = bytearray(SLState.BUFSIZE)
        # AJL packed_buf = [str() for __idx0 in range(BUFSIZE)]
        self.packed_buf = bytearray(SLState.BUFSIZE)
        self.recptr = 0
        self.sendptr = 0
        self.expect_info = False
        self.netto_trig = -1
        self.netdly_trig = 0
        self.keepalive_trig = -1
        self.previous_time = 0.0
        self.netto_time = 0.0
        self.netdly_time = 0.0
        self.keepalive_time = 0.0

    def getPacket(self):
        """
        Returns last received packet.

        :return: last received packet if data buffer contains a full packet to
            send.
        :raise SeedLinkException: if there is not a packet ready to send.

        See also: :meth:`packetAvailable`
        """
        if not self.packetAvailable():
            raise SeedLinkException("SLPacket not available to send")
        return SLPacket(self.databuf, self.sendptr)

    def packetAvailable(self):
        """
        Check for full packet available to send.

        :return: true if data buffer contains a full packet to send.

        See also: :meth:`getPacket`

        """
        return self.recptr - self.sendptr >= \
            SLPacket.SLHEADSIZE + SLPacket.SLRECSIZE

    def bytesRemaining(self):
        """
        Return number of bytes remaining in receiving buffer.

        :return: number of bytes remaining.

        """
        return self.BUFSIZE - self.recptr

    def isError(self):
        """
        Check for SeedLink ERROR packet.

        :return: true if next send packet is a SeedLink ERROR packet

        :raise SeedLinkException: if there are not enough bytes to determine

        """
        if self.recptr - self.sendptr < len(SLPacket.ERRORSIGNATURE):
            msg = "not enough bytes to determine packet type"
            raise SeedLinkException(msg)
        return self.databuf[self.sendptr: self.sendptr +
                            len(SLPacket.ERRORSIGNATURE)].lower() == \
            SLPacket.ERRORSIGNATURE.lower()  # @UndefinedVariable

    def isEnd(self):
        """
        Check for SeedLink END packet.

        :return: true if next send packet is a SeedLink END packet

        :raise SeedLinkException: if there are not enough bytes to determine
        """
        if self.recptr - self.sendptr < len(SLPacket.ENDSIGNATURE):
            msg = "not enough bytes to determine packet type"
            raise SeedLinkException(msg)
        return self.databuf[self.sendptr: self.sendptr +
                            len(SLPacket.ENDSIGNATURE)].lower() == \
            SLPacket.ENDSIGNATURE.lower()  # @UndefinedVariable

    def packetIsInfo(self):
        """
        Check for SeedLink INFO packet.

        :return: true if next send packet is a SeedLink INFO packet

        :raise SeedLinkException: if there are not enough bytes to determine
            packet type
        """
        if self.recptr - self.sendptr < len(SLPacket.INFOSIGNATURE):
            msg = "not enough bytes to determine packet type"
            raise SeedLinkException(msg)
        return self.databuf[self.sendptr: self.sendptr +
                            len(SLPacket.INFOSIGNATURE)].lower() == \
            SLPacket.INFOSIGNATURE.lower()  # @UndefinedVariable

    def incrementSendPointer(self):
        """
        Increments the send pointer by size of one packet.

        """
        self.sendptr += SLPacket.SLHEADSIZE + SLPacket.SLRECSIZE

    def packDataBuffer(self):
        """
        Packs the buffer by removing all sent packets and shifting remaining
        bytes to beginning of buffer.
        """
        # AJL System.arraycopy(self.databuf, self.sendptr, self.packed_buf, 0,
        #                     self.recptr - self.sendptr)
        self.packed_buf[0:self.recptr - self.sendptr] = \
            self.databuf[self.sendptr: self.recptr]
        temp_buf = self.databuf
        self.databuf = self.packed_buf
        self.packed_buf = temp_buf
        self.recptr -= self.sendptr
        self.sendptr = 0

    def appendBytes(self, bytes_):
        """
        Appends bytes to the receive buffer after the last received data.
        """
        if self.bytesRemaining() < len(bytes_):
            msg = "not enough bytes remaining in buffer to append new bytes"
            raise SeedLinkException(msg)

        self.databuf[self.recptr:self.recptr + len(bytes_)] = bytes_
        self.recptr += len(bytes_)
