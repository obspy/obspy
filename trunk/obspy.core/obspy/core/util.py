# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from math import sqrt, pi, sin, cos, asin, tan, atan, atan2
from pkg_resources import require, iter_entry_points
import ctypes as C
import doctest
import functools
import glob
import numpy as np
import os
import sys
import tempfile
import unittest
import warnings


# defining ObsPy modules currently used by runtests and the path function
DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh', 'segy', 'taup']
ALL_MODULES = DEFAULT_MODULES + ['fissures', 'arclink', 'seishub', 'iris',
                                 'neries', 'db']

# default order of automatic format detection
WAVEFORM_PREFERRED_ORDER = ['MSEED', 'SAC', 'GSE2', 'SEISAN', 'SACXY', 'GSE1',
                            'Q', 'SH_ASC', 'SLIST', 'TSPAIR', 'SEGY', 'SU',
                            'WAV']

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


def guessDelta(channel):
    """
    Estimate time delta in seconds between each sample from given channel name.

    :type channel: str
    :param channel: Channel name, e.g. ``'BHZ'`` or ``'H'``
    :rtype: float
    :return: Returns ``0`` if band code is not given or unknown.

    .. rubric:: Example

    >>> print guessDelta('BHZ')
    0.1

    >>> print guessDelta('H')
    0.0125

    >>> print guessDelta('XZY')  # doctest: +SKIP
    0
    """
    try:
        return 1. / BAND_CODE[channel[0]]
    except:
        msg = "No or unknown channel id provided. Specifying a channel id " + \
              "could lead to better selection of first/last samples of " + \
              "fetched traces."
        warnings.warn(msg)
    return 0


class AttribDict(dict, object):
    """
    A class which behaves like a dictionary.

    :type data: dict, optional
    :param data: Dictionary with initial keywords.

    .. rubric:: Basic Usage

    You may use the following syntax to change or access data in this class.

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
    >>> x = sorted(x)
    >>> x[0:3]
    ['network', 'station']
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

    def pop(self, name, default={}):
        value = super(AttribDict, self).pop(name, default)
        del self.__dict__[name]
        return value

    def popitem(self):
        (name, value) = super(AttribDict, self).popitem()
        super(AttribDict, self).__delattr__(name)
        return (name, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, pickle_dict):
        self.update(pickle_dict)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def copy(self):
        return self.__class__(self.__dict__.copy())

    def __deepcopy__(self, *args, **kwargs):  # @UnusedVariable
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

    This function is taken from :func:`scipy.stats.scoreatpercentile`
    Copyright (c) Gary Strangman
    """
    if issorted:
        values = sorted(a)
        if limit:
            values = values[(limit[0] < a) & (a < limit[1])]
    else:
        values = a

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


# C file pointer/ descriptor class
class FILE(C.Structure):  # Never directly used
    """
    C file pointer class for type checking with argtypes
    """
    pass
c_file_p = C.POINTER(FILE)


def formatScientific(value):
    """
    Returns a float string in a fixed exponential style.

    :type value: float
    :param value: Floating point number
    :rtype: str
    :return: Fixed string of given float number.

    Different operation systems are delivering different output for the
    exponential format of floats.

    (1) **Python 2.5.2** (r252:60911, Feb 21 2008, 13:11:45)
        [MSC v.1310 32 bit (Intel)] on **win32**

        >>> '%E' % 2.5 # doctest: +SKIP
        '2.500000E+000'`

    (2) **Python 2.5.2** (r252:60911, Apr  2 2008, 18:38:52)
        [GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on **linux2**

        >>> '%E' % 2.5 # doctest: +SKIP
        '2.500000E+00'

    This function ensures a valid format independent of the operation system.
    For speed issues any number ending with `E+0XX` or `E-0XX` is simply cut
    down to `E+XX` or `E-XX`. This will fail for numbers `XX>99`.

    .. rubric:: Example

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
    Weak replacement for the Python's tempfile.TemporaryFile.

    This function is a replacment for :func:`tempfile.NamedTemporaryFile` but
    will work also with Windows 7/Vista's UAC.

    :type dir: str
    :param dir: If specified, the file will be created in that directory,
        otherwise the default directory for temporary files is used.
    :type suffix: str
    :param suffix: The temporary file name will end with that suffix. Defaults
        to ``'.tmp'``.

    .. warning::
        Caller is responsible for deleting the file when done with it.

    .. rubric:: Example

    >>> ntf = NamedTemporaryFile()
    >>> ntf._fileobj  # doctest: +ELLIPSIS
    <open file '<fdopen>', mode 'w+b' at 0x...>
    >>> ntf._fileobj.close()
    >>> os.remove(ntf.name)

    >>> filename = NamedTemporaryFile().name
    >>> fh = open(filename, 'wb')
    >>> fh.write("test")
    >>> fh.close()
    >>> os.remove(filename)
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

    :type line: str
    :param line: String in the form ``"(real, imag)"``.
    :rtype: complex
    :return: Complex number.

    .. rubric:: Example

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

    :param delta: Number of samples for data chunk
    :param dtype: NumPy dtype for returned data chunk
    :param fill_value: If ``None``, masked array is returned, else the
        array is filled with the corresponding value

    .. rubric:: Example

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
    elif (isinstance(fill_value, list) or isinstance(fill_value, tuple)) \
            and len(fill_value) == 2:
        # if two values are supplied use these as samples bordering to our data
        # and interpolate between:
        ls = fill_value[0]
        rs = fill_value[1]
        # include left and right sample (delta + 2)
        interpolation = np.linspace(ls, rs, delta + 2)
        # cut ls and rs and ensure correct data type
        temp = np.require(interpolation[1:-1], dtype=np.dtype(dtype))
    else:
        temp = np.ones(delta, dtype=np.dtype(dtype))
        temp *= fill_value
    return temp


def getExampleFile(filename):
    """
    Function to find the absolute path of a test data file

    The ObsPy modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    installed ObsPy modules and checks weather the file is in any of
    the "tests/data" subdirectories.

    :param filename: A test file name to which the path should be returned.
    :return: Full path to file.

    .. rubric:: Example

    >>> getExampleFile('slist.ascii')  # doctest: +SKIP
    /custom/path/to/obspy/core/tests/data/slist.ascii

    >>> getExampleFile('does.not.exists')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IOError: Could not find file does.not.exists ...
    """
    for module in ALL_MODULES:
        try:
            mod = __import__("obspy.%s.tests" % module, fromlist=["obspy"])
            file = os.path.join(mod.__path__[0], "data", filename)
            if os.path.isfile(file):
                return file
        except ImportError:
            pass
    msg = "Could not find file %s in tests/data directory " % filename + \
          "of ObsPy modules"
    raise IOError(msg)


def _getVersionString(module="obspy.core"):
    """
    Returns either the .egg version or current SVN revision for a given module.

    .. rubric:: Example

    >>> _getVersionString('obspy.core')  # doctest: +SKIP
    '0.4.8.dev-r2767'

    >>> _getVersionString('does.not.exist')  # doctest: +ELLIPSIS
    'Module does.not.exist is not installed via setup.py!'
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

    :type group: str
    :param group: Group name.
    :type subgroup_name: str, optional
    :param subgroup_name: Subgroup name (defaults to None).
    :rtype: dict
    :returns: Dictionary of entry points of each plug-in.

    .. rubric:: Example

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


def deprecated(func, warning_msg=None):
    """
    This is a decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if 'deprecated' in str(func.__doc__).lower():
            msg = func.__doc__
        elif warning_msg:
            msg = warning_msg
        else:
            msg = "Call to deprecated function %s." % func.__name__
        warnings.warn(msg, category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def deprecated_keywords(keywords):
    def fdec(func):
        fname = func.func_name
        msg = "Deprecated keyword %s in %s() call - please use %s instead."

        @functools.wraps(func)
        def echo_func(*args, **kwargs):
            for kw in kwargs.keys():
                if kw in keywords:
                    nkw = keywords[kw]
                    warnings.warn(msg % (kw, fname, nkw),
                                  category=DeprecationWarning)
                    kwargs[nkw] = kwargs[kw]
                    del(kwargs[kw])
            return func(*args, **kwargs)
        return echo_func

    return fdec


def interceptDict(func):
    """
    This is a decorator to intercept convenience method calls of the old,
    deprecated style (e.g. trace.filter("lowpass", {'freq': 10})).

    For the convenience functions on trace/stream the options that got passed
    on had to be specified as a dictionary in the first implementation and
    now are expected to be given as kwargs directly.
    So we do:

    * throw a DeprecationWarning
    * make the correct call
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # function itself is first arg so len(args) == 3 means we got 2 args...
        if len(args) == 3 and isinstance(args[2], dict):
            msg = "Using a dictionary to pass on filter options will be " + \
                  "removed in the future. Please specify all options as " + \
                  "kwargs."
            warnings.warn(msg, DeprecationWarning)
            kwargs = args[2]
            args = args[:-1]
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

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: str
    :param module_name: name of the module of which the tests should be added

    .. rubric:: Example

    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_doctests(suite, "obspy.core")
    """
    MODULE_NAME = module_name
    MODULE = __import__(MODULE_NAME, fromlist="obspy")

    filename_pattern = os.path.join(MODULE.__path__[0], "*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([MODULE_NAME, name]) for name in names)
    for module_name in module_names:
        module = __import__(module_name, fromlist="obspy")
        testsuite.addTest(doctest.DocTestSuite(module))


def add_unittests(testsuite, module_name):
    """
    Function to add all available unittests of the module with given name
    (e.g. "obspy.core") to the given unittest TestSuite.
    All submodules in the "tests" directory whose names are starting with
    ``test_`` are added.

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: str
    :param module_name: name of the module of which the tests should be added

    .. rubric:: Example

    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_unittests(suite, "obspy.core")
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


def skip(reason):
    """
    Unconditionally skip a test.
    """
    def decorator(test_item):
        if not (isinstance(test_item, type) and issubclass(test_item,
                                                           unittest.TestCase)):
            @functools.wraps(test_item)
            def skip_wrapper(*args, **kwargs):  # @UnusedVariable
                return

            test_item = skip_wrapper

        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = reason
        return test_item
    return decorator


def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)

    def _id(obj):
        return obj

    return _id


def uncompressFile(func):
    """
    Decorator used for temporary uncompressing file if .gz or .bz2 archive.
    """
    def wrapped_func(filename, *args, **kwargs):
        if isinstance(filename, basestring) and not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        # check if we got a compressed file
        unpacked_data = None
        if filename.endswith('.bz2'):
            # bzip2
            try:
                import bz2
                unpacked_data = bz2.decompress(open(filename, 'rb').read())
            except:
                pass
        elif filename.endswith('.gz'):
            # gzip
            try:
                import gzip
                unpacked_data = gzip.open(filename, 'rb').read()
            except:
                pass
        if unpacked_data:
            # we unpacked something without errors - create temporary file
            tempfile = NamedTemporaryFile()
            tempfile._fileobj.write(unpacked_data)
            tempfile.close()
            # call wrapped function
            try:
                result = func(tempfile.name, *args, **kwargs)
            except:
                # clean up unpacking procedure
                if unpacked_data:
                    tempfile.close()
                    os.remove(tempfile.name)
                raise
            # clean up unpacking procedure
            if unpacked_data:
                tempfile.close()
                os.remove(tempfile.name)
        else:
            # call wrapped function with original filename
            result = func(filename, *args, **kwargs)
        return result
    return wrapped_func


def getEntryPoints():
    """
    Creates a sorted list of available entry points.
    """
    # get all available entry points
    formats_ep = _getPlugins('obspy.plugin.waveform', 'readFormat')
    # NOTE: If no file format is installed, this will fail and therefore the
    # whole file can no longer be executed. However obspy.core.ascii is
    # always available.
    if not formats_ep:
        msg = "Your current ObsPy installation does not support any file " + \
              "reading formats. Please update or extend your ObsPy " + \
              "installation."
        raise Exception(msg)
    eps = formats_ep.values()
    names = [_i.name for _i in eps]
    # loop through known waveform plug-ins and add them to resulting list
    new_entries = []
    for entry in WAVEFORM_PREFERRED_ORDER:
        # skip plug-ins which are not installed
        if not entry in names:
            continue
        new_entries.append(formats_ep[entry])
        index = names.index(entry)
        eps.pop(index)
        names.pop(index)
    # extend resulting list with any modules which are unknown
    new_entries.extend(eps)
    # return list of entry points
    return new_entries


def getMatplotlibVersion():
    """
    Get matplotlib version information.

    :returns: Matplotlib version as a list of three integers or ``None`` if
        matplotlib import fails.
    """
    try:
        import matplotlib
        version = matplotlib.__version__.replace('svn', '')
        version = map(int, version.split("."))
    except ImportError:
        version = None
    return version


def _vulnerable_gps2DistAzimuth(lat1, lon1, lat2, lon2):
    """
    For the documentation see :func:`gps2DistAzimuth`

    This method is vulnerable if the two points are close to being antipodes.
    (Starts failing at e.g. (0,0,0,179.4))
    """
    # Check inputs
    if lat1 > 90 or lat1 < -90:
        msg = "Latitude of Point 1 out of bounds! (-90 <= lat1 <=90)"
        raise ValueError(msg)
    while lon1 > 180:
        lon1 -= 360
    while lon1 < -180:
        lon1 += 360
    if lat2 > 90 or lat2 < -90:
        msg = "Latitude of Point 2 out of bounds! (-90 <= lat2 <=90)"
        raise ValueError(msg)
    while lon2 > 180:
        lon2 -= 360
    while lon2 < -180:
        lon2 += 360

    # Data on the WGS84 reference ellipsoid:
    a = 6378137.0          # semimajor axis in m
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)        # semiminor axis

    if (abs(lat1 - lat2) < 1e-8) and (abs(lon1 - lon2) < 1e-8):
        return 0.0, 0.0, 0.0

    # convert latitudes and longitudes to radians:
    lat1 = lat1 * 2.0 * pi / 360.
    lon1 = lon1 * 2.0 * pi / 360.
    lat2 = lat2 * 2.0 * pi / 360.
    lon2 = lon2 * 2.0 * pi / 360.

    TanU1 = (1 - f) * tan(lat1)
    TanU2 = (1 - f) * tan(lat2)

    U1 = atan(TanU1)
    U2 = atan(TanU2)

    dlon = lon2 - lon1
    last_dlon = -4000000.0                # an impossible value
    omega = dlon

    # Iterate until there is no significant change in dlon
    while (last_dlon < -3000000.0 or dlon != 0 and
           abs((last_dlon - dlon) / dlon) > 1.0e-9):
        sqr_sin_sigma = pow(cos(U2) * sin(dlon), 2) + \
            pow((cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(dlon)), 2)
        Sin_sigma = sqrt(sqr_sin_sigma)
        Cos_sigma = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(dlon)
        sigma = atan2(Sin_sigma, Cos_sigma)
        Sin_alpha = cos(U1) * cos(U2) * sin(dlon) / sin(sigma)
        alpha = asin(Sin_alpha)
        Cos2sigma_m = cos(sigma) - (2 * sin(U1) * sin(U2) / pow(cos(alpha), 2))
        C = (f / 16) * pow(cos(alpha), 2) * \
            (4 + f * (4 - 3 * pow(cos(alpha), 2)))
        last_dlon = dlon
        dlon = omega + (1 - C) * f * sin(alpha) * (sigma + C * sin(sigma) * \
            (Cos2sigma_m + C * cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2))))

        u2 = pow(cos(alpha), 2) * (a * a - b * b) / (b * b)
        A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B / 4) * (Cos_sigma * \
            (-1 + 2 * pow(Cos2sigma_m, 2)) - (B / 6) * Cos2sigma_m * \
            (-3 + 4 * sqr_sin_sigma) * (-3 + 4 * pow(Cos2sigma_m, 2))))

        dist = b * A * (sigma - delta_sigma)
        alpha12 = atan2((cos(U2) * sin(dlon)),
                        (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(dlon)))
        alpha21 = atan2((cos(U1) * sin(dlon)),
                        (-sin(U1) * cos(U2) + cos(U1) * sin(U2) * cos(dlon)))

    if alpha12 < 0.0:
        alpha12 = alpha12 + (2.0 * pi)
    if alpha12 > (2.0 * pi):
        alpha12 = alpha12 - (2.0 * pi)

    alpha21 = alpha21 + pi

    if alpha21 < 0.0:
        alpha21 = alpha21 + (2.0 * pi)
    if alpha21 > (2.0 * pi):
        alpha21 = alpha21 - (2.0 * pi)

    # convert to degrees:
    alpha12 = alpha12 * 360 / (2.0 * pi)
    alpha21 = alpha21 * 360 / (2.0 * pi)

    return dist, alpha12, alpha21


def gps2DistAzimuth(lat1, lon1, lat2, lon2):
    """
    Computes the distance between two geographic points on the WGS84
    ellipsoid and the forward and backward azimuths between these points.

    Latitudes should be positive for eastern/northern hemispheres and
    negative for western/southern hemispheres respectively.

    This code is based on an implementation incorporated in
    Matplotlib Basemap Toolkit 0.9.5
    http://sourceforge.net/projects/matplotlib/files/
    (basemap-0.9.5/lib/matplotlib/toolkits/basemap/greatcircle.py)

    Algorithm from Geocentric Datum of Australia Technical Manual.
    http://www.icsm.gov.au/gda/gdatm/index.html
    http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf

    It states::

        Computations on the Ellipsoid

        There are a number of formulae that are available to calculate accurate
        geodetic positions, azimuths and distances on the ellipsoid.

        Vincenty's formulae (Vincenty, 1975) may be used for lines ranging from
        a few cm to nearly 20,000 km, with millimetre accuracy. The formulae
        have been extensively tested for the Australian region, by comparison
        with results from other formulae (Rainsford, 1955 & Sodano, 1965).

        * Inverse problem: azimuth and distance from known latitudes and
            longitudes
        * Direct problem: Latitude and longitude from known position, azimuth
            and distance.

    :param lat1: Latitude of point A in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon1: Longitude of point A in degrees (positive for eastern,
        negative for western hemisphere)
    :param lat2: Latitude of point B in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon2: Longitude of point B in degrees (positive for eastern,
        negative for western hemisphere)
    :return: (Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees)
    """
    try:
        values = _vulnerable_gps2DistAzimuth(lat1, lon1, lat2, lon2)
        if np.alltrue(np.isnan(values)):
            raise ValueError("excepting nan return values")
        return values
    # we should use an alternative calculation method for this case
    # but for now just settle with this quick fix
    # see #150
    except ValueError, e:
        msg = "Catching unstable calculation on antipodes. " + \
              "If this happens too often please bully the developers " + \
              "into implementing a more secure solution for this issue."
        unstable = abs(lon1 - lon2) > 179.3
        if str(e) == "math domain error" and unstable:
            warnings.warn(msg)
            return (20004314.5, 0.0, 0.0)
        elif str(e) == "excepting nan return values" and unstable:
            warnings.warn(msg)
            return (20004314.5, 0.0, 0.0)
        else:
            raise e


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
