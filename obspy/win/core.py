# -*- coding: utf-8 -*-
"""
WIN bindings to ObsPy core module.
"""

from obspy import Trace, UTCDateTime, Stream
import numpy as np
import warnings


def isWIN(filename,century="20"):  # @UnusedVariable
    """
    Checks whether a file is WIN or not.

    :type filename: string
    :param filename: WIN file to be checked.
    :rtype: bool
    :return: ``True`` if a WIN file.
    """
    # as long we don't have full format description we just try to read the
    # file like readWIN and check for errors
    try:
        fpin = open(filename, "rb")
        pklen = fpin.read(4)
        _truelen = np.fromstring(pklen, '>i')[0]  # equiv to Str4Int
        buff = fpin.read(6)
        yy = "%s%02x" % (century, ord(buff[0]))
        mm = "%x" % ord(buff[1])
        dd = "%x" % ord(buff[2])
        hh = "%x" % ord(buff[3])
        mi = "%x" % ord(buff[4])
        sec = "%x" % ord(buff[5])
        # This will raise for invalid dates.
        UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                           int(sec))
        buff = fpin.read(4)
        '%02x' % ord(buff[0])
        '%02x' % ord(buff[1])
        int('%x'% (ord(buff[2]) >> 4))
        ord(buff[3])
        idata00 = fpin.read(4)
        np.fromstring(idata00, '>i')[0]
    except:
        return False
    return True


def readWIN(filename, century="20", **kwargs):  # @UnusedVariable
    """
    Reads a WIN file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: WIN file to be read.
    :param century: WIN stores year as 2 numbers, need century to
        construct proper datetime.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """
    output = {}
    srates = {}

    # read WIN file
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

            yy = "%s%02x" % (century, ord(buff[0]))
            mm = "%x" % ord(buff[1])
            dd = "%x" % ord(buff[2])
            hh = "%x" % ord(buff[3])
            mi = "%x" % ord(buff[4])
            sec = "%x" % ord(buff[5])
            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                               int(sec))
            # print "Second Start Date:", date
            if start == 0:
                start = date
            if status0 == 0:
                sdata = None
            while leng < truelen:
                buff = fpin.read(4)
                leng += 4
                flag = '%02x' % ord(buff[0])
                chanum = '%02x' % ord(buff[1])
                chanum = "%02s%02s"%(flag,chanum)
                datawide = int('%x'% (ord(buff[2]) >> 4))
                srate = ord(buff[3])
                xlen = (srate - 1) * datawide
                if datawide == 0:
                    xlen = srate/2
                    datawide = 0.5

                idata00 = fpin.read(4)
                leng += 4
                idata22 = np.fromstring(idata00, '>i')[0]
                
                if chanum in output:
                    output[chanum].append(idata22)
                else:
                    output[chanum] = [idata22, ]
                    srates[chanum] = srate
                sdata = fpin.read(xlen)
                leng += xlen

                if len(sdata) < xlen:
                    fpin.seek(-(xlen - len(sdata)), 1)
                    sdata += fpin.read(xlen - len(sdata))
                    msg = "This shouldn't happen, it's weird..."
                    warnings.warn(msg)
                
                if datawide == 0.5:
                    for i in range(srate/2):
                        idata2 = output[chanum][-1] + np.fromstring(sdata[i:i + 1], 'b')[0] >> 4
                        output[chanum].append(idata2)
                        idata2 = idata2 + (np.fromstring(sdata[i:i + 1], 'b')[0] << 4) >> 4
                        output[chanum].append(idata2)
                elif datawide == 1:
                    for i in range((xlen / datawide)):
                        idata2 = output[chanum][-1] + np.fromstring(sdata[i:i + 1], 'b')[0]
                        output[chanum].append(idata2)
                elif datawide == 2:
                    for i in range((xlen / datawide)):
                        idata2 = output[chanum][-1] + np.fromstring(sdata[2 * i:2 * (i + 1)],
                                           '>h')[0]
                        output[chanum].append(idata2)
                elif datawide == 3:
                    for i in range((xlen / datawide)):
                        idata2 = output[chanum][-1] + np.fromstring(sdata[3 * i:3 * (i + 1)] + ' ',
                                           '>i')[0] >> 8
                        output[chanum].append(idata2)
                elif datawide == 4:
                    for i in range((xlen / datawide)):
                        idata2 = output[chanum][-1] + np.fromstring(sdata[4 * i:4 * (i + 1)],
                                           '>i')[0]
                        output[chanum].append(idata2)
                else:
                    msg = "DATAWIDE is %s " % datawide + \
                          "but only values of 1, 2, 3 or 4 are supported."
                    raise NotImplementedError(msg)
                

    traces = []
    for i in output.keys():
        t = Trace(data=np.array(output[i]))
        t.stats.channel = str(i)
        t.stats.sampling_rate = float(srates[i])
        t.stats.starttime = start
        traces.append(t)
    return Stream(traces=traces)
