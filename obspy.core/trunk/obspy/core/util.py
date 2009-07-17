# -*- coding: utf-8 -*-

import ctypes as C
import traceback


def getFormatsAndMethods(verbose=False):
    """
    Collects all obspy parser classes.

    @type verbose: Bool
    @param verbose: Print error messages / exceptions while parsing.
    """
    temp = []
    failure = []
    # There is one try-except block for each supported file format.
    try:
        from obspy.mseed.core import isMSEED, readMSEED, writeMSEED
        # The first item is the name of the format, the second the checking function.
        temp.append(['MSEED', isMSEED, readMSEED, writeMSEED])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.gse2.core import isGSE2, readGSE2, writeGSE2
        # The first item is the name of the format, the second the checking function.
        temp.append(['GSE2', isGSE2, readGSE2, writeGSE2])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.wav.core import isWAV, readWAV, writeWAV
        # The first item is the name of the format, the second the checking function.
        temp.append(['WAV', isWAV, readWAV, writeWAV])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.sac.core import isSAC, readSAC, writeSAC
        # The first item is the name of the format, the second the checking function.
        temp.append(['SAC', isSAC, readSAC, writeSAC])
    except:
        failure.append(traceback.format_exc())
    if verbose:
        for _i in xrange(len(failure)):
            print failure[_i]
    return temp


def supportedFormats():
    """
    Returns a list of all file formats supported by ObsPy.
    """
    return [_i[0] for _i in getFormatsAndMethods()]


def scoreatpercentile(a, per, limit=(), sort=True):
    """
    Calculates the score at the given 'per' percentile of the sequence a.

    For example, the score at per = 50 is the median.

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter 'limit' is provided, it should be a tuple (lower,
    upper) of two values.  Values of 'a' outside this (closed) interval
    will be ignored.

        >>> a = [1, 2, 3, 4]
        >>> scoreatpercentile(a, 25)
        1.75
        >>> scoreatpercentile(a, 50)
        2.5
        >>> scoreatpercentile(a, 75)
        3.25
        >>> a = [6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36]
        >>> scoreatpercentile(a, 25)
        25.5
        >>> scoreatpercentile(a, 50)
        40
        >>> scoreatpercentile(a, 75)
        42.5

    This method is taken from scipy.stats.scoreatpercentile
    Copyright (c) Gary Strangman
    """
    if sort:
        values = sorted(a)
        if limit:
            values = values[(limit[0] < a) & (a < limit[1])]
    else:
        values = a

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction;

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


# C file pointer/ descriptor class
class FILE(C.Structure): # Never directly used
    """C file pointer class for type checking with argtypes"""
    pass
c_file_p = C.POINTER(FILE)

# Define ctypes arg- and restypes.
#C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
#C.pythonapi.PyFile_AsFile.restype = c_file_p


def formatScientific(value):
    """
    Formats floats in a fixed exponential format.
    
    Different operation systems are delivering different output for the
    exponential format of floats. Here we ensure to deliver in a for SEED
    valid format independent of the OS. For speed issues we simple cut any 
    number ending with E+0XX or E-0XX down to E+XX or E-XX. This fails for 
    numbers XX>99, but should not occur, because the SEED standard does 
    not allow this values either.
    
    Python 2.5.2 (r252:60911, Feb 21 2008, 13:11:45) 
    [MSC v.1310 32 bit (Intel)] on win32
    > '%E' % 2.5
    '2.500000E+000'
    
    Python 2.5.2 (r252:60911, Apr  2 2008, 18:38:52)
    [GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on linux2
    > '%E' % 2.5
    '2.500000E+00'

    Doctest:
    >>> formatScientific("3.4e+002")
    '3.4e+02'
    >>> formatScientific("3.4E+02")
    '3.4E+02'
    """
    if 'e' in value:
        mantissa, exponent = value.split('e')
        return "%se%+03d" % (mantissa, int(exponent))
    elif 'E' in value:
        mantissa, exponent = value.split('E')
        return "%sE%+03d" % (mantissa, int(exponent))
    else:
        msg = "Can't format scientific %s" % (value)
        raise TypeError(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
