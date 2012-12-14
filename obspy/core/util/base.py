# -*- coding: utf-8 -*-
"""
Base utilities and constants for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.misc import toIntOrZero
from obspy.core.util.types import OrderedDict
from obspy.core.util.version import get_git_version as _getVersionString
from pkg_resources import require, iter_entry_points, load_entry_point
import ctypes as C
import doctest
import glob
import numpy as np
import os
import sys
import tempfile


# defining ObsPy modules currently used by runtests and the path function
DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh', 'segy', 'taup', 'seg2', 'db',
                   'realtime', 'datamark']
NETWORK_MODULES = ['arclink', 'seishub', 'iris', 'neries', 'earthworm',
                   'seedlink']
ALL_MODULES = DEFAULT_MODULES + NETWORK_MODULES

# default order of automatic format detection
WAVEFORM_PREFERRED_ORDER = ['MSEED', 'SAC', 'GSE2', 'SEISAN', 'SACXY', 'GSE1',
                            'Q', 'SH_ASC', 'SLIST', 'TSPAIR', 'SEGY', 'SU',
                            'SEG2', 'WAV', 'PICKLE', 'DATAMARK']

_sys_is_le = sys.byteorder == 'little'
NATIVE_BYTEORDER = _sys_is_le and '<' or '>'


# C file pointer/ descriptor class
class FILE(C.Structure):  # Never directly used
    """
    C file pointer class for type checking with argtypes
    """
    pass
c_file_p = C.POINTER(FILE)


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
    installed ObsPy modules and checks whether the file is in any of
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
        mod = __import__("obspy.%s.tests" % module, fromlist=["obspy"])
        file = os.path.join(mod.__path__[0], "data", filename)
        if os.path.isfile(file):
            return file
    msg = "Could not find file %s in tests/data directory " % filename + \
          "of ObsPy modules"
    raise IOError(msg)


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
    MODULE_PATH = MODULE.__path__[0]
    MODULE_PATH_LEN = len(MODULE_PATH)

    for root, _dirs, files in os.walk(MODULE_PATH):
        # skip directories without __init__.py
        if not '__init__.py' in files:
            continue
        # skip tests directories
        if root.endswith('tests'):
            continue
        # skip scripts directories
        if root.endswith('scripts'):
            continue
        # skip lib directories
        if root.endswith('lib'):
            continue
        # loop over all files
        for file in files:
            # skip if not python source file
            if not file.endswith('.py'):
                continue
            # get module name
            parts = root[MODULE_PATH_LEN:].split(os.sep)[1:]
            module_name = ".".join([MODULE_NAME] + parts + [file[:-3]])
            try:
                module = __import__(module_name, fromlist="obspy")
                testsuite.addTest(doctest.DocTestSuite(module))
            except ValueError:
                pass


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


def _getEntryPoints(group, subgroup=None):
    """
    Gets a dictionary of all available plug-ins of a group or subgroup.

    :type group: str
    :param group: Group name.
    :type subgroup: str, optional
    :param subgroup: Subgroup name (defaults to None).
    :rtype: dict
    :returns: Dictionary of entry points of each plug-in.

    .. rubric:: Example

    >>> _getEntryPoints('obspy.plugin.waveform')  # doctest: +ELLIPSIS
    {...'SLIST': EntryPoint.parse('SLIST = obspy.core.ascii')...}
    """
    features = {}
    for ep in iter_entry_points(group):
        if subgroup:
            if list(iter_entry_points(group + '.' + ep.name, subgroup)):
                features[ep.name] = ep
        else:
            features[ep.name] = ep
    return features


def _getOrderedEntryPoints(group, subgroup=None, order_list=[]):
    """
    Gets a ordered dictionary of all available plug-ins of a group or subgroup.
    """
    # get all available entry points
    ep_dict = _getEntryPoints(group, subgroup)
    # loop through official supported waveform plug-ins and add them to
    # ordered dict of entry points
    entry_points = OrderedDict()
    for name in order_list:
        try:
            entry_points[name] = ep_dict.pop(name)
        except:
            # skip plug-ins which are not installed
            continue
    # extend entry points with any left over waveform plug-ins
    entry_points.update(ep_dict)
    return entry_points


ENTRY_POINTS = {
    'trigger': _getEntryPoints('obspy.plugin.trigger'),
    'filter': _getEntryPoints('obspy.plugin.filter'),
    'rotate': _getEntryPoints('obspy.plugin.rotate'),
    'detrend': _getEntryPoints('obspy.plugin.detrend'),
    'integrate': _getEntryPoints('obspy.plugin.integrate'),
    'differentiate': _getEntryPoints('obspy.plugin.differentiate'),
    'waveform': _getOrderedEntryPoints('obspy.plugin.waveform',
                                       'readFormat', WAVEFORM_PREFERRED_ORDER),
    'waveform_write': _getOrderedEntryPoints('obspy.plugin.waveform',
                                      'writeFormat', WAVEFORM_PREFERRED_ORDER),
    'event': _getEntryPoints('obspy.plugin.event', 'readFormat'),
    'taper': _getEntryPoints('obspy.plugin.taper'),
}


def _getFunctionFromEntryPoint(group, type):
    """
    A "automagic" function searching a given dict of entry points for a valid
    entry point and returns the function call. Otherwise it will raise a
    default error message.

    .. rubric:: Example

    >>> _getFunctionFromEntryPoint('detrend', 'simple')  # doctest: +ELLIPSIS
    <function simple at 0x...>

    >>> _getFunctionFromEntryPoint('detrend', 'XXX')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Detrend type "XXX" is not supported. Supported types: ...
    """
    ep_dict = ENTRY_POINTS[group]
    try:
        # get entry point
        if type in ep_dict:
            entry_point = ep_dict[type]
        else:
            # search using lower cases only
            entry_point = [v for k, v in ep_dict.items()
                           if k.lower() == type.lower()][0]
    except (KeyError, IndexError):
        # check if any entry points are available at all
        if not ep_dict:
            msg = "Your current ObsPy installation does not support " + \
                  "any %s functions. Please make sure " + \
                  "SciPy is installed properly."
            raise ImportError(msg % (group.capitalize()))
        # ok we have entry points, but specified function is not supported
        msg = "%s type \"%s\" is not supported. Supported types: %s"
        raise ValueError(msg % (group.capitalize(), type, ', '.join(ep_dict)))
    # import function point
    # any issue during import of entry point should be raised, so the user has
    # a chance to correct the problem
    func = load_entry_point(entry_point.dist.key,
            'obspy.plugin.%s' % (group), entry_point.name)
    return func


def getMatplotlibVersion():
    """
    Get matplotlib version information.

    :returns: Matplotlib version as a list of three integers or ``None`` if
        matplotlib import fails.
        The last version number can indicate different things like it being a
        version from the old svn trunk, the latest git repo, some release
        candidate version, ...
        If the last number cannot be converted to an integer it will be set to
        0.
    """
    try:
        import matplotlib
        version = matplotlib.__version__
        version = version.split("~rc")[0]
        version = map(toIntOrZero, version.split("."))
    except ImportError:
        version = None
    return version


def _readFromPlugin(plugin_type, filename, format=None, **kwargs):
    """
    Reads a single file from a plug-in's readFormat function.
    """
    EPS = ENTRY_POINTS[plugin_type]
    # get format entry point
    format_ep = None
    if not format:
        # auto detect format - go through all known formats in given sort order
        for format_ep in EPS.values():
            # search isFormat for given entry point
            isFormat = load_entry_point(format_ep.dist.key,
                'obspy.plugin.%s.%s' % (plugin_type, format_ep.name),
                'isFormat')
            # check format
            if isFormat(filename):
                break
        else:
            raise TypeError('Unknown format for file %s' % filename)
    else:
        # format given via argument
        format = format.upper()
        try:
            format_ep = EPS[format]
        except IndexError:
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(EPS)))
    # file format should be known by now
    try:
        # search readFormat for given entry point
        readFormat = load_entry_point(format_ep.dist.key,
            'obspy.plugin.%s.%s' % (plugin_type, format_ep.name), 'readFormat')
    except ImportError:
        msg = "Format \"%s\" is not supported. Supported types: %s"
        raise TypeError(msg % (format_ep.name, ', '.join(EPS)))
    # read
    list_obj = readFormat(filename, **kwargs)
    return list_obj, format_ep.name


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
