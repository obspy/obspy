# -*- coding: utf-8 -*-
"""
Base utilities and constants for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future import standard_library
from future.utils import native_str

import doctest
import inspect
import io
import os
import sys
import tempfile

with standard_library.hooks():
    from collections import OrderedDict

from pkg_resources import iter_entry_points, load_entry_point
import numpy as np

from obspy.core.util.misc import toIntOrZero


# defining ObsPy modules currently used by runtests and the path function
DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh', 'segy', 'taup', 'seg2', 'db',
                   'realtime', 'datamark', 'css', 'y', 'pde', 'station',
                   'ndk', 'ah', 'zmap', 'nlloc', 'pdas', 'cnv', 'kinemetrics']
NETWORK_MODULES = ['arclink', 'seishub', 'iris', 'neries', 'earthworm',
                   'seedlink', 'neic', 'fdsn']
ALL_MODULES = DEFAULT_MODULES + NETWORK_MODULES

# default order of automatic format detection
WAVEFORM_PREFERRED_ORDER = ['MSEED', 'SAC', 'GSE2', 'SEISAN', 'SACXY', 'GSE1',
                            'Q', 'SH_ASC', 'SLIST', 'TSPAIR', 'Y', 'PICKLE',
                            'SEGY', 'SU', 'SEG2', 'WAV', 'DATAMARK', 'CSS',
                            'AH', 'PDAS', 'KINEMETRICS_EVT']
EVENT_PREFERRED_ORDER = ['QUAKEML', 'NLLOC_HYP']

_sys_is_le = sys.byteorder == 'little'
NATIVE_BYTEORDER = _sys_is_le and '<' or '>'


class NamedTemporaryFile(io.BufferedIOBase):
    """
    Weak replacement for the Python's tempfile.TemporaryFile.

    This class is a replacement for :func:`tempfile.NamedTemporaryFile` but
    will work also with Windows 7/Vista's UAC.

    :type dir: str
    :param dir: If specified, the file will be created in that directory,
        otherwise the default directory for temporary files is used.
    :type suffix: str
    :param suffix: The temporary file name will end with that suffix. Defaults
        to ``'.tmp'``.

    .. rubric:: Example

    >>> with NamedTemporaryFile() as tf:
    ...     _ = tf.write(b"test")
    ...     os.path.exists(tf.name)
    True
    >>> # when using the with statement, the file is deleted at the end:
    >>> os.path.exists(tf.name)
    False

    >>> with NamedTemporaryFile() as tf:
    ...     filename = tf.name
    ...     with open(filename, 'wb') as fh:
    ...         _ = fh.write(b"just a test")
    ...     with open(filename, 'r') as fh:
    ...         print(fh.read())
    just a test
    >>> # when using the with statement, the file is deleted at the end:
    >>> os.path.exists(tf.name)
    False
    """
    def __init__(self, dir=None, suffix='.tmp', prefix='obspy-'):
        fd, self.name = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)
        self._fileobj = os.fdopen(fd, 'w+b', 0)  # 0 -> do not buffer

    def read(self, *args, **kwargs):
        return self._fileobj.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self._fileobj.write(*args, **kwargs)

    def seek(self, *args, **kwargs):
        self._fileobj.seek(*args, **kwargs)
        return self._fileobj.tell()

    def tell(self, *args, **kwargs):
        return self._fileobj.tell(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        self.close()  # flush internal buffer
        self._fileobj.close()
        os.remove(self.name)


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

    >>> createEmptyDataChunk(6, np.complex128, 0)
    array([ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])

    >>> createEmptyDataChunk(3, 'f') # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    masked_array(data = [-- -- --],
                 mask = ...,
                 ...)
    """
    # For compatibility with NumPy 1.4
    if isinstance(dtype, str):
        dtype = native_str(dtype)
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
    Function to find the absolute path of a data file

    The ObsPy modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    installed ObsPy modules and checks whether the file is in any of
    the "tests/data/" or "data/" subdirectories.

    :param filename: A test file name to which the path should be returned.
    :return: Full path to file.

    .. rubric:: Example

    >>> getExampleFile('slist.ascii')  # doctest: +SKIP
    /custom/path/to/obspy/core/tests/data/slist.ascii

    >>> getExampleFile('does.not.exists')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    OSError: Could not find file does.not.exists ...
    """
    for module in ALL_MODULES:
        try:
            mod = __import__("obspy.%s" % module,
                             fromlist=[native_str("obspy")])
        except ImportError:
            continue
        file_ = os.path.join(mod.__path__[0], "tests", "data", filename)
        if os.path.isfile(file_):
            return file_
        file_ = os.path.join(mod.__path__[0], "data", filename)
        if os.path.isfile(file_):
            return file_
    msg = ("Could not find file %s in tests/data or data "
           "directory of ObsPy modules") % filename
    raise OSError(msg)


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
    'interpolate': _getEntryPoints('obspy.plugin.interpolate'),
    'integrate': _getEntryPoints('obspy.plugin.integrate'),
    'differentiate': _getEntryPoints('obspy.plugin.differentiate'),
    'waveform': _getOrderedEntryPoints('obspy.plugin.waveform',
                                       'readFormat', WAVEFORM_PREFERRED_ORDER),
    'waveform_write': _getOrderedEntryPoints(
        'obspy.plugin.waveform', 'writeFormat', WAVEFORM_PREFERRED_ORDER),
    'event': _getEntryPoints('obspy.plugin.event', 'readFormat'),
    'event_write': _getEntryPoints('obspy.plugin.event', 'writeFormat'),
    'taper': _getEntryPoints('obspy.plugin.taper'),
    'inventory': _getEntryPoints('obspy.plugin.inventory', 'readFormat'),
    'inventory_write': _getEntryPoints('obspy.plugin.inventory',
                                       'writeFormat'),
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
    func = load_entry_point(entry_point.dist.key, 'obspy.plugin.%s' % (group),
                            entry_point.name)
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
        version = version.split("rc")[0].strip("~")
        version = list(map(toIntOrZero, version.split(".")))
    except ImportError:
        version = None
    return version


def getBasemapVersion():
    """
    Get basemap version information.

    :returns: basemap version as a list of three integers or ``None`` if
        basemap import fails.
        The last version number can indicate different things like it being a
        version from the old svn trunk, the latest git repo, some release
        candidate version, ...
        If the last number cannot be converted to an integer it will be set to
        0.
    """
    try:
        from mpl_toolkits import basemap
        version = basemap.__version__
        version = version.split("rc")[0].strip("~")
        version = list(map(toIntOrZero, version.split(".")))
    except ImportError:
        version = None
    return version


def getSciPyVersion():
    """
    Get SciPy version information.

    :returns: SciPy version as a list of three integers or ``None`` if scipy
        import fails.
        The last version number can indicate different things like it being a
        version from the old svn trunk, the latest git repo, some release
        candidate version, ...
        If the last number cannot be converted to an integer it will be set to
        0.
    """
    try:
        import scipy
        version = scipy.__version__
        version = version.split("~rc")[0]
        version = list(map(toIntOrZero, version.split(".")))
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
            isFormat = load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.%s.%s' % (plugin_type, format_ep.name),
                'isFormat')
            # If it is a file-like object, store the position and restore it
            # later to avoid that the isFormat() functions move the file
            # pointer.
            if hasattr(filename, "tell") and hasattr(filename, "seek"):
                position = filename.tell()
            else:
                position = None
            # check format
            is_format = isFormat(filename)
            if position is not None:
                filename.seek(0, 0)
            if is_format:
                break
        else:
            raise TypeError('Unknown format for file %s' % filename)
    else:
        # format given via argument
        format = format.upper()
        try:
            format_ep = EPS[format]
        except (KeyError, IndexError):
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(EPS)))
    # file format should be known by now
    try:
        # search readFormat for given entry point
        readFormat = load_entry_point(
            format_ep.dist.key,
            'obspy.plugin.%s.%s' % (plugin_type, format_ep.name), 'readFormat')
    except ImportError:
        msg = "Format \"%s\" is not supported. Supported types: %s"
        raise TypeError(msg % (format_ep.name, ', '.join(EPS)))
    # read
    list_obj = readFormat(filename, **kwargs)
    return list_obj, format_ep.name


def getScriptDirName():
    """
    Get the directory of the current script file. This is more robust than
    using __file__.
    """
    return os.path.abspath(os.path.dirname(inspect.getfile(
        inspect.currentframe())))


def make_format_plugin_table(group="waveform", method="read", numspaces=4,
                             unindent_first_line=True):
    """
    Returns a markdown formatted table with read waveform plugins to insert
    in docstrings.

    >>> table = make_format_plugin_table("event", "write", 4, True)
    >>> print(table)  # doctest: +NORMALIZE_WHITESPACE
    ========= ================== ========================================
        Format    Required Module    _`Linked Function Call`
        ========= ================== ========================================
        CNV       :mod:`obspy.cnv`   :func:`obspy.cnv.core.write_CNV`
        JSON      :mod:`obspy.core`  :func:`obspy.core.json.core.writeJSON`
        NLLOC_OBS :mod:`obspy.nlloc` :func:`obspy.nlloc.core.write_nlloc_obs`
        QUAKEML   :mod:`obspy.core`  :func:`obspy.core.quakeml.writeQuakeML`
        ZMAP      :mod:`obspy.zmap`  :func:`obspy.zmap.core.writeZmap`
        ========= ================== ========================================

    :type group: str
    :param group: Plugin group to search (e.g. "waveform" or "event").
    :type method: str
    :param method: Either 'read' or 'write' to select plugins based on either
        read or write capability.
    :type numspaces: int
    :param numspaces: Number of spaces prepended to each line (for indentation
        in docstrings).
    :type unindent_first_line: bool
    :param unindent_first_line: Determines if first line should start with
        prepended spaces or not.
    """
    method = method.lower()
    if method not in ("read", "write"):
        raise ValueError("no valid type: %s" % method)

    method += "Format"
    eps = _getOrderedEntryPoints("obspy.plugin.%s" % group, method,
                                 WAVEFORM_PREFERRED_ORDER)
    mod_list = []
    for name, ep in eps.items():
        module_short = ":mod:`%s`" % ".".join(ep.module_name.split(".")[:2])
        func = load_entry_point(ep.dist.key,
                                "obspy.plugin.%s.%s" % (group, name), method)
        func_str = ':func:`%s`' % ".".join((ep.module_name, func.__name__))
        mod_list.append((name, module_short, func_str))

    mod_list = sorted(mod_list)
    headers = ["Format", "Required Module", "_`Linked Function Call`"]
    maxlens = [max([len(x[0]) for x in mod_list] + [len(headers[0])]),
               max([len(x[1]) for x in mod_list] + [len(headers[1])]),
               max([len(x[2]) for x in mod_list] + [len(headers[2])])]

    info_str = [" ".join(["=" * x for x in maxlens])]
    info_str.append(
        " ".join([headers[i].ljust(maxlens[i]) for i in range(3)]))
    info_str.append(info_str[0])

    for mod_infos in mod_list:
        info_str.append(
            " ".join([mod_infos[i].ljust(maxlens[i]) for i in range(3)]))
    info_str.append(info_str[0])

    ret = " " * numspaces + ("\n" + " " * numspaces).join(info_str)
    if unindent_first_line:
        ret = ret[numspaces:]
    return ret


class ComparingObject(object):
    """
    Simple base class that implements == and != based on self.__dict__
    """
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)


def _DeprecatedArgumentAction(old_name, new_name, real_action='store'):
    """
    Specifies deprecated command-line arguments to scripts
    """
    message = '%s has been deprecated. Please use %s in the future.' % (
        old_name, new_name
    )

    from argparse import Action

    class _Action(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            import warnings
            warnings.warn(message)

            # I wish there were an easier way...
            if real_action == 'store':
                setattr(namespace, self.dest, values)
            elif real_action == 'store_true':
                setattr(namespace, self.dest, True)
            elif real_action == 'store_false':
                setattr(namespace, self.dest, False)

    return _Action


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
