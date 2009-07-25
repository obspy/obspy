# -*- coding: utf-8 -*-

from obspy.core import Trace, UTCDateTime
from obspy.sac.sacio import ReadSac
import numpy as N
import array, os


def isSAC(filename):
    """
    Checks whether a file is SAC or not. Returns True or False.
    
    @param filename: SAC file to be read.
    """
    g = ReadSac()
    try:
        npts = g.GetHvalueFromFile(filename, 'npts')
    except:
        return False
    st = os.stat(filename) #file's size = st[6] 
    sizecheck = st[6] - (632 + 4 * npts)
    # size check info
    if sizecheck != 0:
        # File-size and theoretical size inconsistent!
        return False
    return True


# we put here everything but the time, they are going to starttime
# left sac attributes, right trace attributes
#XXX NOTE not all values from the read in dictonary are converted
# this is definetly a problem when reading an writing a read sac file.
convert_dict = {'npts': 'npts',
                'delta':'sampling_rate',
                'kcmpnm': 'channel',
                'kstnm': 'station'
}

sac_extra = ['depmin', 'depmax', 'scale', 'odelta', 'b', 'e', 'o', 'a',
             't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9',
             'f', 'stla', 'stlo', 'stel', 'stdp', 'evla', 'evlo', 'evdp',
             'mag', 'user0', 'user1', 'user2', 'user3', 'user4', 'user5',
             'user6', 'user7', 'user8', 'user9', 'dist', 'az', 'baz',
             'gcarc', 'depmen', 'cmpaz', 'cmpinc', 'nzyear', 'nzjday',
             'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'nvhdr', 'norid',
             'nevid', 'nwfid', 'iftype', 'idep', 'iztype', 'iinst',
             'istreg', 'ievreg', 'ievtype', 'iqual', 'isynth', 'imagtyp',
             'imagsrc', 'leven', 'lpspol', 'lovrok', 'lcalda', 'kevnm',
             'khole', 'ko', 'ka', 'kt0', 'kt1', 'kt2', 'kt3', 'kt4', 'kt5',
             'kt6', 'kt7', 'kt8', 'kt9', 'kf', 'kuser0', 'kuser1',
             'kuser2', 'knetwk', 'kdatrd', 'kinst',
]

def readSAC(filename, headonly=False, **kwargs):
    """
    Reads a SAC file and returns an obspy.Trace object.
    
    @param filename: SAC file to be read.
    """
    # read SAC file
    t = ReadSac()
    if headonly:
        t.ReadSacHeader(filename)
    else:
        t.ReadSacFile(filename)
    # assign all header entries to a new dictionary compatible with an Obspy
    header = {}
    for i, j in convert_dict.iteritems():
        header[j] = t.GetHvalue(i)
    header['sac'] = {}
    for i in sac_extra:
        header['sac'][i] = t.GetHvalue(i)
    # convert time to UTCDateTime
    year = t.GetHvalue('nzyear')
    yday = t.GetHvalue('nzjday')
    hour = t.GetHvalue('nzhour')
    mint = t.GetHvalue('nzmin')
    sec = t.GetHvalue('nzsec')
    msec = t.GetHvalue('nzmsec')
    microsec = msec * 1000
    mon, day = UTCDateTime.strptime(str(year) + str(yday), "%Y%j").timetuple()[1:3]
    header['starttime'] = UTCDateTime(year, mon, day, hour, mint, sec, microsec)
    header['endtime'] = UTCDateTime(header['starttime'].timestamp +
        header['npts'] / float(header['sampling_rate'])
    )
    if headonly:
        return Trace(header=header)
    #XXX From Python2.6 the buffer interface can be generally used to
    # directly pass the pointers from the array.array class to
    # numpy.ndarray, old version:
    # data=N.fromstring(t.seis.tostring(),dtype='float32'))
    return Trace(header=header,
                 data=N.frombuffer(t.seis, dtype='float32'))


def writeSAC(stream_object, filename, **kwargs):
    """
    Write SAC file and returns an obspy.Trace object.
    
    @param filename: SAC file to be read.
    """
    #
    # Translate the common (renamed) entries
    i = 0
    for trace in stream_object:
        t = ReadSac()
        t.InitArrays() # initialize header arrays
        #
        # Check for necessary values, set a default if they are missing
        trace.stats.setdefault('npts', len(trace.data)) # set the number of data points
        trace.stats.setdefault('sampling_rate', 1.0)
        trace.stats.setdefault('starttime', UTCDateTime(0.0))
        trace.stats.sac.setdefault('nvhdr', 1)  # SAC version needed 0<version<20
        #
        for _j, _k in convert_dict.iteritems():
            try:
                t.SetHvalue(_j, trace.stats[_k])
            except:
                pass
        # filling up from the sac specific part
        for _i in sac_extra:
            try:
                t.SetHvalue(_i, trace.stats.sac[_i])
            except:
                pass
        # year, month, day, hour, min, sec
        try:
            (year, month, day, hour, mint,
                    sec) = trace.stats.starttime.timetuple()[0:6]
            msec = trace.stats.starttime.microsecond / 1e3
            yday = trace.stats.starttime.strftime("%j")
            t.SetHvalue('nzyear', year)
            t.SetHvalue('nzjday', yday)
            t.SetHvalue('nzhour', hour)
            t.SetHvalue('nzmin', mint)
            t.SetHvalue('nzsec', sec)
            t.SetHvalue('nzmsec', msec)
        except:
            raise
        if trace.data.dtype != 'float32':
            trace.data = trace.data.astype('float32')
        # building array of floats
        t.seis = array.array('f')
        # pass data as string (actually it's a copy), using a list for
        # passing would require a type info per list entry and thus use a lot
        # of memory
        # XXX use the buffer interface at soon as it is supported in
        # array.array, Python2.6
        t.seis.fromstring(trace.data.tostring())
        if i != 0:
            base ,ext = os.path.splitext(filename)
            filename = "%s%02d%s" % (base,i,ext)
        try:
            t.WriteSacBinary(filename)
        except:
            raise
        i += 1

