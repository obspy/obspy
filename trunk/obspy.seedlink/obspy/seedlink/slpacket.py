# -*- coding: utf-8 -*-
"""
Module to hold and decode a SeedLink packet.

Part of Python implementaion of libslink of Chad Trabant and
JSeedLink of Anthony Lomax

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


from obspy.core.trace import Trace
from obspy.mseed import LibMSEED
from obspy.mseed.headers import HPTMODULUS, \
        MSRecord
from obspy.mseed.libmseed import clibmseed
import ctypes as C
import numpy as np

class SLPacket(object):
    """ 
    Class to hold and decode a SeedLink packet.
    
    :var TYPE_SLINFT: Packet type is terminated info packet.
    :type TYPE_SLINFT: int
    
    :var TYPE_SLINF: Packet type is non-terminated info packet.
    :type TYPE_SLINF: int
    
    :var SLTERMINATE: Terminate flag - connection was closed by the server or the 
    termination sequence completed.
    :type SLTERMINATE: str
    
    :var SLNOPACKET: No packet flag - indicates no data availbale.
    :type SLNOPACKET: chr
    
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
    
    :var msrecord: The mini-SEED record.
    :type msrecord: bytes
    
    :var blockette: The Blockette conained in msrecord.
    :type blockette: list

    """

    TYPE_SLINFT = -101
    TYPE_SLINF = -102
    SLTERMINATE = "SLTERMINATE"
    SLNOPACKET = "SLNOPACKET"
    SLERROR = "SLERROR"
    SLHEADSIZE = 8
    SLRECSIZE = 512
    SIGNATURE = "SL"
    INFOSIGNATURE = "SLINFO"
    ERRORSIGNATURE = "ERROR\r\n"
    ENDSIGNATURE = "END"
    slhead = None
    msrecord = None
    blockette = None
    trace = None


    def __init__(self, bytes=None, offset=None):
        if bytes is None or offset is None:
            return
        if len(bytes) - offset < self.SLHEADSIZE + self.SLRECSIZE:
            raise SeedLinkException("not enough bytes in subarray to construct a new SLPacket")
        self.slhead = bytes[offset : offset + self.SLHEADSIZE]
        self.msrecord = bytes[offset + self.SLHEADSIZE : offset + self.SLHEADSIZE + self.SLRECSIZE]

    def getSequenceNumber(self):
        #print "DEBUG: repr(self.slhead):", repr(self.slhead)
        #print "DEBUG: self.slhead[0 : len(self.INFOSIGNATURE)].lower():", self.slhead[0 : len(self.INFOSIGNATURE)].lower()
        #print "DEBUG: self.INFOSIGNATURE.lower():", self.INFOSIGNATURE.lower()
        if self.slhead[0 : len(self.INFOSIGNATURE)].lower() == self.INFOSIGNATURE.lower():
            return 0
        #print "DEBUG: self.slhead[0 : len(self.SIGNATURE)].lower():", self.slhead[0 : len(self.SIGNATURE)].lower()
        #print "DEBUG: self.SIGNATURE.lower():", self.SIGNATURE.lower()
        if not self.slhead[0 : len(self.SIGNATURE)].lower() == self.SIGNATURE.lower():
            return -1
        seqstr = str(self.slhead[2:8])
        #print "DEBUG: seqstr:", seqstr,", int(seqstr, 16):", int(seqstr, 16)
        seqnum = -1
        try:
            seqnum = int(seqstr, 16)
        except Exception:
            print "SLPacket.getSequenceNumber(): bad packet sequence number: " + seqstr
            return -1
        return seqnum


    def getMSRecord(self):
        # following from  obspy.mseed.tests.test_libmseed.py -> test_msrParse(self)
        msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        pyobj = np.array(self.msrecord)
        errcode = \
                clibmseed.msr_parse(pyobj.ctypes.data_as(C.POINTER(C.c_char)),
                len(pyobj), C.pointer(msr), -1, 1, 1)
        if errcode != 0:
            raise SeedLinkException("failed to decode mini-seed record: msr_parse errcode: " + errcode)
        #print "DEBUG: msr:", msr
        msrecord_py = msr.contents
        #print "DEBUG: msrecord_py:", msrecord_py

        return msrecord_py

    def getTrace(self):

        if self.trace is not None:
            return self.trace

        msrecord_py = self.getMSRecord()
        #print "DEBUG: msrecord_py:", msrecord_py
        header = LibMSEED()._convertMSRToDict(msrecord_py)
        # 20111201 AJL - bug fix?
        header['starttime'] = header['starttime'] / HPTMODULUS
        # 20111205 AJL - bug fix?
        if 'samprate' in header:
            header['sampling_rate'] = header['samprate']
            del header['samprate']
        # Access data directly as NumPy array.
        data = LibMSEED()._ctypesArray2NumpyArray(msrecord_py.datasamples,
                                                  msrecord_py.numsamples,
                                                  msrecord_py.sampletype)
        self.trace = Trace(data, header)
        return self.trace


    def getType(self):
        #print "DEBUG: self.slhead:", repr(self.slhead)
        if self.slhead[0 : len(SLPacket.INFOSIGNATURE)].lower() == SLPacket.INFOSIGNATURE.lower():
            if (chr(self.slhead[self.SLHEADSIZE - 1]) != '*'):
                return self.TYPE_SLINFT
            else:
                return self.TYPE_SLINF
        msrecord_py = self.getMSRecord()
        #print "DEBUG: msrecord_py:", msrecord_py
        #print "DEBUG: msrecord_py.reclen:", msrecord_py.reclen
        #print "DEBUG: msrecord_py.sequence_number:", msrecord_py.sequence_number
        #print "DEBUG: msrecord_py.samplecnt:", msrecord_py.samplecnt
        #print "DEBUG: msrecord_py.encoding:", msrecord_py.encoding
        #print "DEBUG: msrecord_py.byteorder:", msrecord_py.byteorder
        #print "DEBUG: msrecord_py.numsamples:", msrecord_py.numsamples
        #print "DEBUG: msrecord_py.sampletype:", msrecord_py.sampletype
        #print "DEBUG: msrecord_py.blkts:", msrecord_py.blkts
        blockette = msrecord_py.blkts.contents
        while blockette:
            #print "DEBUG: ===================="
            #print "DEBUG: blkt_type:", blockette.blkt_type
            #print "DEBUG: next_blkt:", blockette.next_blkt
            #print "DEBUG: blktdata:", blockette.blktdata
            #print "DEBUG: blktdatalen:", blockette.blktdatalen
            #print "DEBUG: next:", blockette.next
            try:
                blockette = blockette.next.contents
            except:
                blockette = None

        # 20111202 AJL - seems to be a bug in obspy.mseed.headers.py:
        #   blkt_link_s type not same as in libmseed.h
        #   obspy blkt_link_s does not have first member of libmseed.h->blkt_link_s:
        #       uint16_t blktoffset;  /* Offset to this blockette */
        #  so with obspy blkt_link_s, blkt_type is in next_blkt
        #  Fixed: obspy.org Ticket #312 (closed defect: fixed)
        #    TODO: update code when version with fix available
        #print "obspy version:", oc.util._getVersionString(module="obspy.core")
        #return msrecord_py.blkts.contents.blkt_type
        return msrecord_py.blkts.contents.next_blkt


# old SLPacket.java methods that should be replaced ane removed

    def getBlockette(self):
        if self.blockette is not None:
            return self.blockette
        seedBuilder = SeedObjectBuilder()
        seedDirector = SeedImportDirector(seedBuilder)
        try:
            seedIn = DataInputStream(ByteArrayInputStream(self.msrecord))
            seedDirector.construct(seedIn)
            container = seedBuilder.getContainer()
            numElem = container.iterate()
            self.blockette = container.getNext()
            return self.blockette
        except (Exception, ), e:
            raise SeedLinkException("failed to decode mini-seed record: " + e)


