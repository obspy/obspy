# -*- coding: utf-8 -*-
"""
Module to hold and decode a SeedLink packet.

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

from obspy.core.compatibility import frombuffer
from obspy.core.trace import Trace
from obspy.core.util.decorator import deprecated_keywords
from obspy.mseed.headers import clibmseed, HPTMODULUS
from obspy.mseed.util import _convertMSRToDict, _ctypesArray2NumpyArray
from obspy.seedlink.seedlinkexception import SeedLinkException
import ctypes as C
import numpy as np


class SLPacket(object):
    """
    Class to hold and decode a SeedLink packet.

    :var TYPE_SLINFT: Packet type is terminated info packet.
    :type TYPE_SLINFT: int
    :var TYPE_SLINF: Packet type is non-terminated info packet.
    :type TYPE_SLINF: int
    :var SLTERMINATE: Terminate flag - connection was closed by the server or
        the termination sequence completed.
    :type SLTERMINATE: str
    :var SLNOPACKET: No packet flag - indicates no data available.
    :type SLNOPACKET: bytes
    :var SLERROR: Error flag - indicates server reported an error.
    :type SLERROR: str
    :var SLHEADSIZE: SeedLink packet header size.
    :type SLHEADSIZE: int
    :var SLRECSIZE: Mini-SEED record size.
    :type SLRECSIZE: int
    :var SIGNATURE: SeedLink header signature.
    :type SIGNATURE: str
    :var INFOSIGNATURE: SeedLink INFO packet signature.
    :type INFOSIGNATURE: str
    :var ERRORSIGNATURE: SeedLink ERROR signature.
    :type ERRORSIGNATURE: str
    :var ENDSIGNATURE: SeedLink END signature.
    :type ENDSIGNATURE: str
    :var slhead: The SeedLink header.
    :type slhead: bytes
    :var msrecord: The MiniSEED record.
    :type msrecord: bytes
    """
    TYPE_SLINFT = -101
    TYPE_SLINF = -102
    SLTERMINATE = b"SLTERMINATE"
    SLNOPACKET = b"SLNOPACKET"
    SLERROR = b"SLERROR"
    SLHEADSIZE = 8
    SLRECSIZE = 512
    SIGNATURE = b"SL"
    INFOSIGNATURE = b"SLINFO"
    ERRORSIGNATURE = b"ERROR\r\n"
    ENDSIGNATURE = b"END"

    @deprecated_keywords({'bytes': 'data'})
    def __init__(self, data=None, offset=None):
        if data is None or offset is None:
            return
        if len(data) - offset < self.SLHEADSIZE + self.SLRECSIZE:
            msg = "not enough bytes in sub array to construct a new SLPacket"
            raise SeedLinkException(msg)
        self.slhead = data[offset: offset + self.SLHEADSIZE]
        self.msrecord = data[offset + self.SLHEADSIZE:
                             offset + self.SLHEADSIZE + self.SLRECSIZE]
        self.trace = None

    def getSequenceNumber(self):
        # print "DEBUG: repr(self.slhead):", repr(self.slhead)
        # print "DEBUG: self.slhead[0 : len(self.INFOSIGNATURE)].lower():",
        # print self.slhead[0 : len(self.INFOSIGNATURE)].lower()
        # print "DEBUG: self.INFOSIGNATURE.lower():",
        #         self.INFOSIGNATURE.lower()
        if self.slhead[0: len(self.INFOSIGNATURE)].lower() == \
                self.INFOSIGNATURE.lower():
            return 0
        # print "DEBUG: self.slhead[0 : len(self.SIGNATURE)].lower():",
        # print self.slhead[0 : len(self.SIGNATURE)].lower()
        # print "DEBUG: self.SIGNATURE.lower():", self.SIGNATURE.lower()
        if not self.slhead[0: len(self.SIGNATURE)].lower() == \
                self.SIGNATURE.lower():
            return -1
        seqbytes = bytes(self.slhead[2:8])
        # print "DEBUG: seqbytes:", seqbytes,", int(seqbytes, 16):", \
        #      int(seqbytes, 16)
        seqnum = -1
        try:
            seqnum = int(seqbytes, 16)
        except Exception:
            msg = "SLPacket.getSequenceNumber(): bad packet sequence number: "
            print(msg, seqbytes)
            return -1
        return seqnum

    def getMSRecord(self):
        # following from  obspy.mseed.tests.test_libmseed.py -> test_msrParse
        msr = clibmseed.msr_init(None)
        pyobj = frombuffer(self.msrecord, dtype=np.uint8)
        errcode = \
            clibmseed.msr_parse(pyobj.ctypes.data_as(C.POINTER(C.c_char)),
                                len(pyobj), C.pointer(msr), -1, 1, 1)
        if errcode != 0:
            msg = "failed to decode mini-seed record: msr_parse errcode: %s"
            raise SeedLinkException(msg % (errcode))
        # print "DEBUG: msr:", msr
        msrecord_py = msr.contents
        # print "DEBUG: msrecord_py:", msrecord_py
        return msr, msrecord_py

    def freeMSRecord(self, msr, msrecord_py):
        clibmseed.msr_free(msr)

    def getTrace(self):

        if self.trace is not None:
            return self.trace

        msr, msrecord_py = self.getMSRecord()
        try:
            header = _convertMSRToDict(msrecord_py)

            # XXX Workaround: in Python 3 msrecord_py.sampletype is a byte
            # (e.g. b'i'), while keys of mseed.headers.SAMPLESIZES are
            # unicode ('i') (see above)
            sampletype = msrecord_py.sampletype
            if not isinstance(sampletype, str):
                sampletype = sampletype.decode()

            data = _ctypesArray2NumpyArray(msrecord_py.datasamples,
                                           msrecord_py.numsamples,
                                           sampletype)
        finally:
            self.freeMSRecord(msr, msrecord_py)

        # XXX Workaround: the fields in the returned struct of type
        # obspy.mseed.header.MSRecord_s have byte values in Python 3, while
        # the rest of the code still expects them to be string (see #770)
        # -> convert
        convert = ('network', 'station', 'location', 'channel',
                   'dataquality', 'sampletype')
        for key, value in header.items():
            if key in convert and not isinstance(value, str):
                header[key] = value.decode()

        # 20111201 AJL - bug fix?
        header['starttime'] = header['starttime'] / HPTMODULUS
        # 20111205 AJL - bug fix?
        if 'samprate' in header:
            header['sampling_rate'] = header['samprate']
            del header['samprate']
        # Access data directly as NumPy array.

        self.trace = Trace(data, header)
        return self.trace

    def getStringPayload(self):
        """
        Get the MiniSEED payload, parsed as string.
        """
        msr, msrecord_py = self.getMSRecord()

        try:
            # This is the same data buffer that is accessed by
            # _ctypesArray2NumpyArray in getTrace above.
            payload = C.string_at(msrecord_py.datasamples,
                                  msrecord_py.samplecnt)
        finally:
            self.freeMSRecord(msr, msrecord_py)

        return payload

    def getType(self):
        # print "DEBUG: self.slhead:", repr(self.slhead)
        if self.slhead[0: len(SLPacket.INFOSIGNATURE)].lower() == \
                SLPacket.INFOSIGNATURE.lower():
            if (chr(self.slhead[self.SLHEADSIZE - 1]) != '*'):
                return self.TYPE_SLINFT
            else:
                return self.TYPE_SLINF
        msr, msrecord_py = self.getMSRecord()
        try:
            ret = msrecord_py.blkts.contents.blkt_type
        finally:
            self.freeMSRecord(msr, msrecord_py)
        return ret
