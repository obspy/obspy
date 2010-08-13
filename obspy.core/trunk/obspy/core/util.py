# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from pkg_resources import require, iter_entry_points
import ctypes as C
import numpy as np
import os
import sys
import tempfile
import inspect
import warnings
import glob
import doctest


# defining ObsPy modules
# currently used by runtests and the path function
DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh']
ALL_MODULES = DEFAULT_MODULES + ['fissures', 'arclink', 'seishub']


_sys_is_le = sys.byteorder == 'little'
NATIVE_BYTEORDER = _sys_is_le and '<' or '>'

# The following dictionary maps the first character of the channel_id to the
# lowest sampling rate this so called Band Code should be used for according
# to: SEED MANUAL p.124
# We use this e.g. in seihub.client.getWaveform to request two samples more on
# both start and end to cut to the samples that really are nearest to requested
# start/endtime afterwards.
BAND_CODE = {'F': 1000.0,
             'G': 1000.0,
             'D': 250.0,
             'C': 250.0,
             'E': 80.0,
             'S': 10.0,
             'H': 80.0,
             'B': 10.0,
             'M': 1.0,
             'L': 1.0,
             'V': 0.1,
             'U': 0.01,
             'R': 0.0001,
             'P': 0.000001,
             'T': 0.0000001,
             'Q': 0.00000001, }


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
        if name in self.readonly:
            return self.__dict__[name]
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


def scoreatpercentile(a, per, limit=(), issorted=True):
    """ 
    Calculates the score at the given per percentile of the sequence a.

    For example, the score at ``per=50`` is the median. 

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter ``limit`` is provided, it should be a tuple (lower,
    upper) of two values.  Values of ``a`` outside this (closed) interval
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
    if issorted:
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


def createEmptyDataChunk(delta, dtype, fill_value=None):
    """
    Creates an NumPy array depending on the given data type and fill value.

    If no ``fill_value`` is given a masked array will be returned.

    @param delta: Number of samples for data chunk
    @param dtype: NumPy dtype for returned data chunk
    @param fill_value: If None, masked array is returned, if not None the
                       array is filled with the corresponding value

    Basic Usage
    -----------
    >>> createEmptyDataChunk(3, 'int', 10)
    array([10, 10, 10])
    >>> createEmptyDataChunk(6, np.dtype('complex128'), 0)
    array([ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])
    >>> createEmptyDataChunk(3, 'f') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    masked_array(data = [-- -- --],
                 mask = ...,
                 ...)
    """
    if fill_value is None:
        temp = np.ma.masked_all(delta, dtype=np.dtype(dtype))
    else:
        temp = np.ones(delta, dtype=np.dtype(dtype))
        temp *= fill_value
    return temp


def path(testfile):
    """
    Function to find the absolute path of a test data file

    The ObsPy modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    installed obspy modules and checks weather the testfile is in any of
    the "tests/data" subdirectories.

    :param testfile: The testfile to which the path should be returned.
    :return: Full path to testfile.

    >>> path('slist.ascii') # doctest: +SKIP
    /custom/path/to/obspy/core/tests/data/slist.ascii
    """
    for module in ALL_MODULES:
        try:
            mod = __import__("obspy.%s.tests" % module, fromlist=["obspy"])
            dir = os.path.dirname(inspect.getsourcefile(mod))
            file = os.path.join(dir, "data", testfile)
            if os.path.isfile(file):
                return file
        except ImportError:
            pass
    msg = "Could not find file %s in tests/data directory " % testfile + \
          "of obspy modules"
    raise IOError(msg)


def _getVersionString(module="obspy.core"):
    """
    Returns either the EGG version or current SVN revision for a given module.
    """
    try:
        mod = require(module)[0]
    except:
        return "Module %s is not installed via setup.py!" % module
    egg_version = mod.version
    # check installation location for .svn directory
    if '.svn' in os.listdir(mod.location):
        path = os.path.join(mod.location, '.svn', 'entries')
        try:
            svn_version = open(path).readlines()[3].strip()
        except:
            return egg_version
        else:
            temp = egg_version.split('.dev-r')
            return temp[0] + ".dev-r" + svn_version
    # else return egg-info version
    return egg_version


def _getPlugins(group, subgroup_name=None):
    """
    Gets a dictionary of all available waveform features plug-ins.

    Parameter
    ---------
    group : string
        Group name.
    subgroup_name : string, optional
        Subgroup name (defaults to None).

    Returns
    -------
    dict
        Dictionary of entry points of each plug-in.

    Basic Usage
    -----------
    >>> _getPlugins('obspy.plugin.waveform')  # doctest: +SKIP
    {'SAC': EntryPoint.parse('SAC = obspy.sac.core'), 'MSEED': EntryPoint...}
    """
    features = {}
    for ep in iter_entry_points(group):
        if subgroup_name:
            if list(iter_entry_points(group + '.' + ep.name, subgroup_name)):
                features[ep.name] = ep
        else:
            features[ep.name] = ep
    return features


def deprecated(func):
    """
    This is a decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """
    def new_func(*args, **kwargs):
        if 'deprecated' in func.__doc__.lower():
            msg = func.__doc__
        else:
            msg = "Call to deprecated function %s." % func.__name__
        warnings.warn(msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def add_doctests(testsuite, module_name):
    """
    Function to add all available doctests of the module with given name
    (e.g. "obspy.core") to the given unittest TestSuite.
    All submodules in the module's root directory are added.
    Occurring errors are shown as warnings.

    Example
    -------
    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_doctests(suite, "obspy.core")

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: String
    :param module_name: name of the module of which the tests should be added
    """
    MODULE_NAME = module_name
    MODULE = __import__(MODULE_NAME, fromlist="obspy")

    filename_pattern = os.path.join(MODULE.__path__[0], "*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([MODULE_NAME, name]) for name in names)
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist="obspy")
            testsuite.addTest(doctest.DocTestSuite(module))
        except Exception, e:
            warnings.warn(str(e))
            pass


def add_unittests(testsuite, module_name):
    """
    Function to add all available unittests of the module with given name
    (e.g. "obspy.core") to the given unittest TestSuite.
    All submodules in the "tests" directory whose names are starting with
    "test_" are added.

    Example
    -------
    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_unittests(suite, "obspy.core")

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: String
    :param module_name: name of the module of which the tests should be added
    """
    MODULE_NAME = module_name
    MODULE_TESTS = __import__(MODULE_NAME + ".tests", fromlist="obspy")

    filename_pattern = os.path.join(MODULE_TESTS.__path__[0], "test_*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([MODULE_NAME, "tests", name]) for name in names)
    for module_name in module_names:
        module = __import__(module_name, fromlist="obspy")
        testsuite.addTest(module.suite())


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
