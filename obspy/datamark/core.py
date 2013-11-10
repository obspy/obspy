# -*- coding: utf-8 -*-
"""
DATAMARK bindings to ObsPy core module.
"""

from obspy import Trace, UTCDateTime, Stream
import numpy as np
import warnings


def isDATAMARK(filename):  # @UnusedVariable
    """
    Checks whether a file is DATAMARK or not.

    :type filename: string
    :param filename: DATAMARK file to be checked.
    :rtype: bool
    :return: ``True`` if a DATAMARK file.
    """
    # as long we don't have full format description we just try to read the
    # file like readDATAMARK and check for errors
    try:
        fpin = open(filename, "rb")
        fpin.read(4)
        buff = fpin.read(6)
        yy = "%s%02x" % (20, np.fromstring(buff[0], dtype='b')[0])
        mm = "%x" % np.fromstring(buff[1], dtype='b')[0]
        dd = "%x" % np.fromstring(buff[2], dtype='b')[0]
        hh = "%x" % np.fromstring(buff[3], dtype='b')[0]
        mi = "%x" % np.fromstring(buff[4], dtype='b')[0]
        sec = "%x" % np.fromstring(buff[5], dtype='b')[0]

        # This will raise for invalid dates.
        UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                    int(sec))
        buff = fpin.read(4)
        np.fromstring(buff[0], dtype='b')[0]
        np.fromstring(buff[1], dtype='b')[0]
        np.fromstring(buff[2], dtype='b')[0] >> 4
        np.fromstring(buff[3], dtype='b')[0]
        idata00 = fpin.read(4)
        np.fromstring(idata00, '>i')[0]
    except:
        return False
    return True


def readDATAMARK(filename, century="20", **kwargs):  # @UnusedVariable
    """
    Reads a DATAMARK file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: DATAMARK file to be read.
    :param century: DATAMARK stores year as 2 numbers, need century to
        construct proper datetime.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """
    output = {}

    # read datamark file
    with open(filename, "rb") as fpin:
        fpin.seek(0, 2)
        sz = fpin.tell()
        fpin.seek(0)
        leng = 0
        status0 = 0
        start = 0
        while leng < sz:
            pklen = fpin.read(4)
            if len(pklen) == 0:
                break  # EOF
            leng = 4
            truelen = np.fromstring(pklen, '>i')[0]  # equiv to Str4Int
            buff = fpin.read(6)
            leng += 6

            yy = "%s%02x" % (century, np.fromstring(buff[0], dtype='b')[0])
            mm = "%x" % np.fromstring(buff[1], dtype='b')[0]
            dd = "%x" % np.fromstring(buff[2], dtype='b')[0]
            hh = "%x" % np.fromstring(buff[3], dtype='b')[0]
            mi = "%x" % np.fromstring(buff[4], dtype='b')[0]
            sec = "%x" % np.fromstring(buff[5], dtype='b')[0]

            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                               int(sec))
            if start == 0:
                start = date
            if status0 == 0:
                sdata = None
            while leng < truelen:
                buff = fpin.read(4)
                leng += 4
                #_flag = np.fromstring(buff[0], dtype='b')[0]
                chanum = np.fromstring(buff[1], dtype='b')[0]
                datawide = np.fromstring(buff[2], dtype='b')[0] >> 4
                srate = np.fromstring(buff[3], dtype='b')[0]
                xlen = (srate - 1) * datawide
                idata00 = fpin.read(4)
                leng += 4
                idata22 = np.fromstring(idata00, '>i')[0]

                if chanum in output:
                    output[chanum].append(idata22)
                else:
                    output[chanum] = [idata22, ]
                sdata = fpin.read(xlen)
                leng += xlen

                if len(sdata) < xlen:
                    fpin.seek(-(xlen - len(sdata)), 1)
                    sdata += fpin.read(xlen - len(sdata))
                    msg = "This shouldn't happen, it's weird..."
                    warnings.warn(msg)
                for i in range((xlen / datawide)):
                    idata2 = 0
                    if datawide == 1:
                        idata2 = np.fromstring(sdata[i:i + 1], 'b')[0]
                    elif datawide == 2:
                        idata2 = np.fromstring(sdata[2 * i:2 * (i + 1)],
                                               '>h')[0]
                    elif datawide == 3:
                        idata2 = np.fromstring(sdata[3 * i:3 * (i + 1)] + ' ',
                                               '>i')[0] >> 8
                    elif datawide == 4:
                        idata2 = np.fromstring(sdata[4 * i:4 * (i + 1)],
                                               '>i')[0]
                    else:
                        msg = "DATAWIDE is %s " % datawide + \
                              "but only values of 1, 2, 3 or 4 are supported."
                        raise NotImplementedError(msg)
                    idata22 += idata2
                    output[chanum].append(idata22)

    traces = []
    for i in output.keys():
        t = Trace(data=np.array(output[i]))
        t.stats.channel = str(i)
        t.stats.sampling_rate = float(srate)
        t.stats.starttime = start
        traces.append(t)
    return Stream(traces=traces)
