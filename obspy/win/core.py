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
        yy = "%s%x" % (century, ord(buff[0]))
        mm = "%x" % ord(buff[1])
        dd = "%x" % ord(buff[2])
        hh = "%x" % ord(buff[3])
        mi = "%x" % ord(buff[4])
        sec = "%x" % ord(buff[5])

        _date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                           int(sec))
        buff = fpin.read(4)
        _flag = np.fromstring(buff[0], dtype='b')[0]
        _chanum = np.fromstring(buff[1], dtype='b')[0]
        _datawide = np.fromstring(buff[2], dtype='b')[0] >> 4
        _srate = np.fromstring(buff[3], dtype='b')[0]
        idata00 = fpin.read(4)
        _idata22 = np.fromstring(idata00, '>i')[0]
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

            yy = "%s%x" % (century, ord(buff[0]))
            mm = "%x" % ord(buff[1])
            dd = "%x" % ord(buff[2])
            hh = "%x" % ord(buff[3])
            mi = "%x" % ord(buff[4])
            sec = "%x" % ord(buff[5])
            date = UTCDateTime(int(yy), int(mm), int(dd), int(hh), int(mi),
                               int(sec))
            print "Second Start Date:", date
            if start == 0:
                start = date
            if status0 == 0:
                sdata = None
            while leng < truelen:
                buff = fpin.read(4)
                # print leng, truelen, len(buff)
                leng += 4
                _flag = '%02x' % ord(buff[0])
                chanum = '%02x' % ord(buff[1])
                chanum = "%02s%02s"%(_flag,chanum)
                datawide = int('%x'% (ord(buff[2]) >> 4))
                srate = ord(buff[3])
                
                # print 'flag, chanum, brol, datawide, srate',_flag, chanum, np.fromstring(buff[2], dtype='b')[0], datawide, srate

                xlen = (srate - 1) * datawide
                if datawide == 0:
                    xlen = srate/2
                    datawide = 0.5
                # print "xlen:", xlen
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
                
                # print "len(sdata)", len(sdata)
                if datawide == 0.5:
                    # print "TODO"
                    
                    for i in range(srate/2):
                        # print i, len(sdata[i:i+1])
                        idata2 = output[chanum][-1] + np.fromstring(sdata[i:i + 1], 'b')[0] >> 4
                        output[chanum].append(idata2)
                        # print idata2
                        
                        idata2 = idata2 + (np.fromstring(sdata[i:i + 1], 'b')[0] << 4) >> 4
                        output[chanum].append(idata2)
                        # print idata2
                        # for(i=1;i<s_rate;i+=2) {
                            # abuf[i]=abuf[i-1]+((*(char *)dp)>>4);
                            # abuf[i+1]=abuf[i]+(((char)(*(dp++)<<4))>>4);
                elif datawide == 1:
                    # print "range:", xlen/datawide
                    for i in range((xlen / datawide)):
                        #abuf[i]=abuf[i-1]+(*(char *)(dp++));
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
