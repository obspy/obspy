# -*- coding: utf-8 -*-
"""
Low-level Earthworm Wave Server tools.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Victor Kress
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import socket
import struct

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats


RETURNFLAG_KEY = {
    'F': 'success',
    'FR': 'requested data right (later) than tank contents',
    'FL': 'requested data left (earlier) than tank contents',
    'FG': 'requested data lie in tank gap',
    'FB': 'syntax error in request',
    'FC': 'data tank corrupt',
    'FN': 'requested tank not found',
    'FU': 'unknown error'
}

DATATYPE_KEY = {
    't4': '>f4', 't8': '>f8',
    's4': '>i4', 's2': '>i2',
    'f4': '<f4', 'f8': '<f8',
    'i4': '<i4', 'i2': '<i2'
}


def getNumpyType(tpstr):
    """
    given a tracebuf2 type string from header,
    return appropriate numpy.dtype object
    """
    dtypestr = DATATYPE_KEY[tpstr]
    tp = np.dtype(dtypestr)
    return tp


class tracebuf2:
    """
    """
    byteswap = False
    ndata = 0           # number of samples in instance
    inputType = None    # NumPy data type

    def readTB2(self, tb2):
        """
        Reads single tracebuf2 packet from beginning of input byte array tb.
        returns number of bytes read or 0 on read fail.
        """
        if len(tb2) < 64:
            return 0   # not enough array to hold header
        head = tb2[:64]
        self.parseHeader(head)
        nbytes = 64 + self.ndata * self.inputType.itemsize
        if len(tb2) < nbytes:
            return 0   # not enough array to hold data specified in header
        dat = tb2[64:nbytes]
        self.parseData(dat)
        return nbytes

    def parseHeader(self, head):
        """
        Parse tracebuf header into class variables
        """
        packStr = b'2i3d7s9s4s3s2s3s2s2s'
        dtype = head[-7:-5]
        if dtype[0] in 'ts':
            endian = b'>'
        elif dtype[0] in 'if':
            endian = b'<'
        else:
            raise ValueError
        self.inputType = getNumpyType(dtype)
        (self.pinno, self.ndata, ts, te, self.rate, self.sta, self.net,
         self.chan, self.loc, self.version, tp, self.qual, _pad) = \
            struct.unpack(endian + packStr, head)
        if not tp.startswith(dtype):
            print('Error parsing header: %s!=%s' % (dtype, tp))
        self.start = UTCDateTime(ts)
        self.end = UTCDateTime(te)
        return

    def parseData(self, dat):
        """
        Parse tracebuf char array data into self.data
        """
        self.data = np.fromstring(dat, self.inputType)
        ndat = len(self.data)
        if self.ndata != ndat:
            print('data count in header (%d) != data count (%d)' % (self.nsamp,
                                                                    ndat))
            self.ndata = ndat
        return

    def getObspyTrace(self):
        """
        Return class contents as obspy.Trace object
        """
        stat = Stats()
        stat.network = self.net.split(b'\x00')[0].decode()
        stat.station = self.sta.split(b'\x00')[0].decode()
        location = self.loc.split(b'\x00')[0].decode()
        if location == '--':
            stat.location = ''
        else:
            stat.location = location
        stat.channel = self.chan.split(b'\x00')[0].decode()
        stat.starttime = UTCDateTime(self.start)
        stat.sampling_rate = self.rate
        stat.npts = len(self.data)
        return Trace(data=self.data, header=stat)


def sendSockReq(server, port, reqStr, timeout=None):
    """
    Sets up socket to server and port, sends reqStr
    to socket and returns open socket
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((server, port))
    if reqStr[-1] == '\n':
        s.send(reqStr)
    else:
        s.send(reqStr + '\n')
    return s


def getSockCharLine(sock, timeout=10.):
    """
    Retrieves one newline terminated string from input open socket
    """
    sock.settimeout(timeout)
    chunks = []
    indat = b'^'
    try:
        while indat[-1] != b'\n':
            # see http://obspy.org/ticket/383
            # indat = sock.recv(8192)
            indat = sock.recv(1)
            chunks.append(indat)
    except socket.timeout:
        print('socket timeout in getSockCharLine()')
        return None
    if chunks:
        response = b''.join(chunks)
        return response
    else:
        return None


def getSockBytes(sock, nbytes, timeout=None):
    """
    Listens for nbytes from open socket.
    Returns byte array as python string or None if timeout
    """
    sock.settimeout(timeout)
    chunks = []
    btoread = nbytes
    try:
        while btoread:
            indat = sock.recv(min(btoread, 8192))
            btoread -= len(indat)
            chunks.append(indat)
    except socket.timeout:
        print('socket timeout in getSockBytes()')
        return None
    if chunks:
        response = ''.join(chunks)
        return response
    else:
        return None


def getMenu(server, port, scnl=None, timeout=None):
    """
    Return list of tanks on server
    """
    rid = 'getMenu'
    if scnl:
        # only works on regular waveservers (not winston)
        getstr = 'MENUSCNL: %s %s %s %s %s\n' % (
            rid, scnl[0], scnl[1], scnl[2], scnl[3])
    else:
        # added SCNL not documented but required
        getstr = 'MENU: %s SCNL\n' % rid
    sock = sendSockReq(server, port,
                       getstr.encode('ascii', 'strict'),
                       timeout=timeout)
    r = getSockCharLine(sock, timeout=timeout)
    sock.close()
    if r:
        # XXX: we got here from bytes to utf-8 to keep the remaining code
        # intact
        tokens = r.decode().split()
        if tokens[0] == rid:
            tokens = tokens[1:]
        flag = tokens[-1]
        if flag in ['FN', 'FC', 'FU']:
            print('request returned %s - %s' % (flag, RETURNFLAG_KEY[flag]))
            return []
        if tokens[7] in DATATYPE_KEY:
            elen = 8  # length of return entry if location included
        elif tokens[6] in DATATYPE_KEY:
            elen = 7  # length of return entry if location omitted
        else:
            print('no type token found in getMenu')
            return []
        outlist = []
        for p in range(0, len(tokens), elen):
            l = tokens[p:p + elen]
            if elen == 8:
                outlist.append((int(l[0]), l[1], l[2], l[3], l[4],
                                float(l[5]), float(l[6]), l[7]))
            else:
                outlist.append((int(l[0]), l[1], l[2], l[3], '--',
                                float(l[4]), float(l[5]), l[6]))
        return outlist
    return []


def readWaveServerV(server, port, scnl, start, end, timeout=None):
    """
    Reads data for specified time interval and scnl on specified waveserverV.

    Returns list of tracebuf2 objects
    """
    rid = 'rwserv'
    scnlstr = '%s %s %s %s' % scnl
    reqstr = 'GETSCNLRAW: %s %s %f %f\n' % (rid, scnlstr, start, end)
    sock = sendSockReq(server, port, reqstr, timeout=timeout)
    r = getSockCharLine(sock, timeout=timeout)
    if not r:
        return []
    tokens = r.decode().split()
    flag = tokens[6]
    if flag != 'F':
        msg = 'readWaveServerV returned flag %s - %s'
        print(msg % (flag, RETURNFLAG_KEY[flag]))
        return []
    nbytes = int(tokens[-1])
    dat = getSockBytes(sock, nbytes, timeout=timeout)
    sock.close()
    tbl = []
    new = tracebuf2()  # empty..filled below
    bytesread = 1
    p = 0
    while bytesread and p < len(dat):
        bytesread = new.readTB2(dat[p:])
        if bytesread:
            tbl.append(new)
            new = tracebuf2()  # empty..filled on next iteration
            p += bytesread
    return tbl


def tracebufs2obspyStream(tbuflist):
    """
    Returns obspy.Stream object from input list of tracebuf2 objects
    """
    if not tbuflist:
        return None
    tlist = []
    for tb in tbuflist:
        tlist.append(tb.getObspyTrace())
    strm = Stream(tlist)
    return strm
