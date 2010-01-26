# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from math import modf, floor
import ctypes as C
import os
import tempfile


class AttribDict(dict, object):
    """
    A class which behaves like a dictionary.

    Basic Usage
    -----------
    You may use the following syntax to change or access data in this
    class.

    >>> stats = AttribDict()
    >>> stats.network = 'BW'
    >>> stats['station'] = 'ROTZ'
    >>> stats.get('network')
    'BW'
    >>> stats['network']
    'BW'
    >>> stats.station
    'ROTZ'
    >>> x = stats.keys()
    >>> x.sort()
    >>> x[0:3]
    ['network', 'station']

    Parameters
    ----------
    data : dict, optional
        Dictionary with initial keywords.
    """
    readonly = []

    def __init__(self, data={}):
        dict.__init__(data)
        self.update(data)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        super(AttribDict, self).__setattr__(key, value)
        super(AttribDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(AttribDict, self).__getitem__(name)

    def __delitem__(self, name):
        super(AttribDict, self).__delattr__(name)
        return super(AttribDict, self).__delitem__(name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, pickle_dict):
        self.update(pickle_dict)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def copy(self, init={}):
        return self.__class__(init)

    def __deepcopy__(self, *args, **kwargs):
        st = self.__class__()
        st.update(self)
        return st

    def update(self, adict={}):
        for (key, value) in adict.iteritems():
            if key in self.readonly:
                continue
            self[key] = value


def quantile(x, q, qtype=7, issorted=False):
    """
    Compute quantiles from input array using a given algorithm type.

    Parameters
    ----------
    x : array
        Input data.
    q : float, `0.0 <= q <= 1.0`
        Quantile. For median, specify `q=0.5`.
    qtype : int , optional
        Selected algorithm. Defaults to 7.
            ==============  =======================================
            Algorithm Type  Algorithm
            ==============  =======================================
            1               Inverse empirical distribution function
            2               Similar to type 1, averaged
            3               Nearest order statistic, (SAS)
            4               California linear interpolation
            5               Hydrologists method
            6               Mean-based estimate(Weibull method)
            7               Mode-based method (S, S-Plus)
            8               Median-unbiased
            9               Normal-unbiased
            ==============  =======================================
    issorted : boolean, optional
        True if `x` already sorted. Defaults to false.

    Examples
    --------
    >>> a = [1, 2, 3, 4]
    >>> quantile(a, 0.25)
    1.75
    >>> quantile(a, 0.50)
    2.5
    >>> quantile(a, 0.75)
    3.25

    >>> a = [6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36]
    >>> quantile(a, 0.25)
    25.5
    >>> quantile(a, 0.50)
    40
    >>> quantile(a, 0.75)
    42.5

    :Author:
        'Ernesto P.Adorio Ph.D.'_, UP Extension Program in Pampanga, Clark Field
    
..  _'Ernesto P.Adorio Ph.D.': http://adorio-research.org/wordpress/?p=125 
    """
    # sort list
    if not issorted:
        y = sorted(x)
    else:
        y = x
    if not (1 <= qtype <= 9):
        return None  # error!
    # Parameters for the Hyndman and Fan algorithm
    abcd = [
        (0, 0, 1, 0), # inverse empirical distrib.function., R type 1
        (0.5, 0, 1, 0), # similar to type 1, averaged, R type 2
        (0.5, 0, 0, 0), # nearest order statistic,(SAS) R type 3
        (0, 0, 0, 1), # California linear interpolation, R type 4
        (0.5, 0, 0, 1), # hydrologists method, R type 5
        (0, 1, 0, 1), # mean-based estimate(Weibull method), R type 6
        (1, -1, 0, 1), # mode-based method,(S, S-Plus), R type 7
        (1.0 / 3, 1.0 / 3, 0, 1), # median-unbiased ,  R type 8
        (3 / 8.0, 0.25, 0, 1)   # normal-unbiased, R type 9.
    ]
    a, b, c, d = abcd[qtype - 1]
    n = len(x)
    g, j = modf(a + (n + b) * q - 1)
    if j < 0:
        return y[0]
    elif j > n:
        return y[n]
    j = int(floor(j))
    if g == 0:
        return y[j]
    else:
        return y[j] + (y[j + 1] - y[j]) * (c + d * g)


# C file pointer/ descriptor class
class FILE(C.Structure): # Never directly used
    """
    C file pointer class for type checking with argtypes
    """
    pass
c_file_p = C.POINTER(FILE)

# Define ctypes arg- and restypes.
#C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
#C.pythonapi.PyFile_AsFile.restype = c_file_p


def formatScientific(value):
    """
    Returns a float string in a fixed exponential style.

    Different operation systems are delivering different output for the
    exponential format of floats.

    (1) Python 2.5.2, WinXP, 32bit::
        Python 2.5.2 (r252:60911, Feb 21 2008, 13:11:45)
        [MSC v.1310 32 bit (Intel)] on win32

        >>> '%E' % 2.5 # doctest: +SKIP
        '2.500000E+000'`

    (2) **Python 2.5.2** (r252:60911, Apr  2 2008, 18:38:52)
        [GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on **linux2**

        >>> '%E' % 2.5 # doctest: +SKIP
        '2.500000E+00'

    This function ensures a valid format independent of the operation system.
    For speed issues any number ending with `E+0XX` or `E-0XX` is simply cut
    down to `E+XX` or `E-XX`. This will fail for numbers `XX>99`.

    Basic Usage
    -----------
    >>> formatScientific("3.4e+002")
    '3.4e+02'

    >>> formatScientific("3.4E+02")
    '3.4E+02'

    >>> formatScientific("%-10.4e" % 0.5960000)
    '5.9600e-01'
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


def NamedTemporaryFile(dir=None, suffix='.tmp'):
    """
    Weak replacement for the Python class :class:`tempfile.NamedTemporaryFile`.

    This class will work also with Windows Vista's UAC.

    .. warning::
        The calling program is responsible to close the returned file pointer
        after usage.
    """

    class NamedTemporaryFile(object):

        def __init__(self, fd, fname):
            self._fileobj = os.fdopen(fd, 'w+b')
            self.name = fname

        def __getattr__(self, attr):
            return getattr(self._fileobj, attr)
    return NamedTemporaryFile(*tempfile.mkstemp(dir=dir, suffix=suffix))


def complexifyString(line):
    """
    Converts a string in the form "(real, imag)" into a complex type.

    Basic Usage
    -----------
    >>> complexifyString("(1,2)")
    (1+2j)

    >>> complexifyString(" ( 1 , 2 ) ")
    (1+2j)
    """
    temp = line.split(',')
    return complex(float(temp[0].strip()[1:]), float(temp[1].strip()[:-1]))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
