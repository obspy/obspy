# -*- coding: utf-8 -*-
"""
Base utilities and constants for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import glob
import importlib
import inspect
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
import warnings
from collections import OrderedDict
from pathlib import PurePath

import numpy as np
import pkg_resources
from pkg_resources import get_entry_info, iter_entry_points

from obspy.core.util.misc import to_int_or_zero, buffered_load_entry_point


# defining ObsPy modules currently used by runtests and the path function
DEFAULT_MODULES = ['clients.filesystem', 'core', 'geodetics', 'imaging',
                   'io.ah', 'io.alsep', 'io.arclink', 'io.ascii',
                   'io.cmtsolution', 'io.cnv', 'io.css', 'io.dmx', 'io.focmec',
                   'io.hypodd', 'io.iaspei', 'io.gcf', 'io.gse2', 'io.json',
                   'io.kinemetrics', 'io.kml', 'io.mseed', 'io.ndk', 'io.nied',
                   'io.nlloc', 'io.nordic', 'io.pdas', 'io.pde', 'io.quakeml',
                   'io.reftek', 'io.rg16', 'io.sac', 'io.scardec', 'io.seg2',
                   'io.segy', 'io.seisan', 'io.sh', 'io.shapefile',
                   'io.seiscomp', 'io.stationtxt', 'io.stationxml', 'io.wav',
                   'io.win', 'io.xseed', 'io.y', 'io.zmap', 'realtime',
                   'scripts', 'signal', 'taup']
NETWORK_MODULES = ['clients.earthworm', 'clients.fdsn',
                   'clients.iris', 'clients.neic', 'clients.nrl',
                   'clients.seedlink', 'clients.syngine']
ALL_MODULES = DEFAULT_MODULES + NETWORK_MODULES

# default order of automatic format detection
WAVEFORM_PREFERRED_ORDER = ['MSEED', 'SAC', 'GSE2', 'SEISAN', 'SACXY', 'GSE1',
                            'Q', 'SH_ASC', 'SLIST', 'TSPAIR', 'Y', 'PICKLE',
                            'SEGY', 'SU', 'SEG2', 'WAV', 'WIN', 'CSS',
                            'NNSA_KB_CORE', 'AH', 'PDAS', 'KINEMETRICS_EVT',
                            'GCF', 'DMX', 'ALSEP_PSE', 'ALSEP_WTN',
                            'ALSEP_WTH']
EVENT_PREFERRED_ORDER = ['QUAKEML', 'NLLOC_HYP']
INVENTORY_PREFERRED_ORDER = ['STATIONXML', 'SEED', 'RESP']
# waveform plugins accepting a byteorder keyword
WAVEFORM_ACCEPT_BYTEORDER = ['MSEED', 'Q', 'SAC', 'SEGY', 'SU']

_sys_is_le = sys.byteorder == 'little'
NATIVE_BYTEORDER = _sys_is_le and '<' or '>'

# Define Obspy hard and soft dependencies
HARD_DEPENDENCIES = [
    "future", "numpy", "scipy", "matplotlib", "lxml.etree", "setuptools",
    "sqlalchemy", "decorator", "requests"]
OPTIONAL_DEPENDENCIES = [
    "flake8", "pyimgur", "pyproj", "pep8-naming", "m2crypto", "shapefile",
    "mock", "pyflakes", "geographiclib", "cartopy"]
DEPENDENCIES = HARD_DEPENDENCIES + OPTIONAL_DEPENDENCIES


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

    def close(self, *args, **kwargs):
        super(NamedTemporaryFile, self).close(*args, **kwargs)
        self._fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        self.close()
        Path(self.name).unlink()


def create_empty_data_chunk(delta, dtype, fill_value=None):
    """
    Creates an NumPy array depending on the given data type and fill value.

    If no ``fill_value`` is given a masked array will be returned.

    :param delta: Number of samples for data chunk
    :param dtype: NumPy dtype for returned data chunk
    :param fill_value: If ``None``, masked array is returned, else the
        array is filled with the corresponding value

    .. rubric:: Example

    >>> create_empty_data_chunk(3, 'int', 10)
    array([10, 10, 10])

    >>> create_empty_data_chunk(
    ...     3, 'f')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    masked_array(data = [-- -- --],
                 mask = ...,
                 ...)
    """
    if fill_value is None:
        temp = np.ma.masked_all(delta, dtype=np.dtype(dtype))
        # fill with nan if float number and otherwise with a very small number
        if issubclass(temp.data.dtype.type, np.integer):
            temp.data[:] = np.iinfo(temp.data.dtype).min
        else:
            temp.data[:] = np.nan
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


def get_example_file(filename):
    """
    Function to find the absolute path of a data file

    The ObsPy modules are installed to a custom installation directory.
    That is the path cannot be predicted. This functions searches for all
    installed ObsPy modules and checks whether the file is in any of
    the "tests/data/" or "data/" subdirectories.

    :param filename: A test file name to which the path should be returned.
    :return: Full path to file.

    .. rubric:: Example

    >>> get_example_file('slist.ascii')  # doctest: +SKIP
    /custom/path/to/obspy/io/ascii/tests/data/slist.ascii

    >>> get_example_file('does.not.exists')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    OSError: Could not find file does.not.exists ...
    """
    for module in ALL_MODULES:
        try:
            mod = __import__("obspy.%s" % module,
                             fromlist=["obspy"])
        except ImportError:
            continue
        file_ = Path(mod.__path__[0]) / "tests" / "data" / filename
        if file_.is_file():
            return str(file_)
        file_ = Path(mod.__path__[0]) / "data" / filename
        if file_.is_file():
            return str(file_)
    msg = ("Could not find file %s in tests/data or data "
           "directory of ObsPy modules") % filename
    raise OSError(msg)


def _get_entry_points(group, subgroup=None):
    """
    Gets a dictionary of all available plug-ins of a group or subgroup.

    :type group: str
    :param group: Group name.
    :type subgroup: str, optional
    :param subgroup: Subgroup name (defaults to None).
    :rtype: dict
    :returns: Dictionary of entry points of each plug-in.

    .. rubric:: Example

    >>> _get_entry_points('obspy.plugin.waveform')  # doctest: +ELLIPSIS
    {...'SLIST': EntryPoint.parse('SLIST = obspy.io.ascii.core')...}
    """
    features = {}
    for ep in iter_entry_points(group):
        if subgroup:
            if list(iter_entry_points(group + '.' + ep.name, subgroup)):
                features[ep.name] = ep
        else:
            features[ep.name] = ep
    return features


def _get_ordered_entry_points(group, subgroup=None, order_list=[]):
    """
    Gets a ordered dictionary of all available plug-ins of a group or subgroup.
    """
    # get all available entry points
    ep_dict = _get_entry_points(group, subgroup)
    # loop through official supported waveform plug-ins and add them to
    # ordered dict of entry points
    entry_points = OrderedDict()
    for name in order_list:
        try:
            entry_points[name] = ep_dict.pop(name)
        except Exception:
            # skip plug-ins which are not installed
            continue
    # extend entry points with any left over waveform plug-ins
    entry_points.update(ep_dict)
    return entry_points


ENTRY_POINTS = {
    'trigger': _get_entry_points('obspy.plugin.trigger'),
    'filter': _get_entry_points('obspy.plugin.filter'),
    'rotate': _get_entry_points('obspy.plugin.rotate'),
    'detrend': _get_entry_points('obspy.plugin.detrend'),
    'interpolate': _get_entry_points('obspy.plugin.interpolate'),
    'integrate': _get_entry_points('obspy.plugin.integrate'),
    'differentiate': _get_entry_points('obspy.plugin.differentiate'),
    'waveform': _get_ordered_entry_points(
        'obspy.plugin.waveform', 'readFormat', WAVEFORM_PREFERRED_ORDER),
    'waveform_write': _get_ordered_entry_points(
        'obspy.plugin.waveform', 'writeFormat', WAVEFORM_PREFERRED_ORDER),
    'event': _get_ordered_entry_points('obspy.plugin.event', 'readFormat',
                                       EVENT_PREFERRED_ORDER),
    'event_write': _get_entry_points('obspy.plugin.event', 'writeFormat'),
    'taper': _get_entry_points('obspy.plugin.taper'),
    'inventory': _get_ordered_entry_points(
        'obspy.plugin.inventory', 'readFormat', INVENTORY_PREFERRED_ORDER),
    'inventory_write': _get_entry_points(
        'obspy.plugin.inventory', 'writeFormat'),
}


def _get_function_from_entry_point(group, type):
    """
    A "automagic" function searching a given dict of entry points for a valid
    entry point and returns the function call. Otherwise it will raise a
    default error message.

    .. rubric:: Example

    >>> _get_function_from_entry_point(
    ...     'detrend', 'simple')  # doctest: +ELLIPSIS
    <function simple at 0x...>

    >>> _get_function_from_entry_point('detrend', 'XXX')  # doctest: +ELLIPSIS
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
    func = buffered_load_entry_point(entry_point.dist.key,
                                     'obspy.plugin.%s' % (group),
                                     entry_point.name)
    return func


def get_dependency_version(package_name, raw_string=False):
    """
    Get version information of a dependency package.

    :type package_name: str
    :param package_name: Name of package to return version info for
    :returns: Package version as a list of three integers or ``None`` if
        import fails. With option ``raw_string=True`` returns raw version
        string instead (or ``None`` if import fails).
        The last version number can indicate different things like it being a
        version from the old svn trunk, the latest git repo, some release
        candidate version, ...
        If the last number cannot be converted to an integer it will be set to
        0.
    """
    try:
        version_string = pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return []
    if raw_string:
        return version_string
    version_list = version_string.split("rc")[0].strip("~")
    version_list = list(map(to_int_or_zero, version_list.split(".")))
    return version_list


NUMPY_VERSION = get_dependency_version('numpy')
SCIPY_VERSION = get_dependency_version('scipy')
MATPLOTLIB_VERSION = get_dependency_version('matplotlib')
CARTOPY_VERSION = get_dependency_version('cartopy')


def _read_from_plugin(plugin_type, filename, format=None, **kwargs):
    """
    Reads a single file from a plug-in's readFormat function.
    """
    if isinstance(filename, str):
        if not Path(filename).exists():
            msg = "[Errno 2] No such file or directory: '{}'".format(
                filename)
            raise FileNotFoundError(msg)
    eps = ENTRY_POINTS[plugin_type]
    # get format entry point
    format_ep = None
    if not format:
        # auto detect format - go through all known formats in given sort order
        for format_ep in eps.values():
            # search isFormat for given entry point
            is_format = buffered_load_entry_point(
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
            is_format = is_format(filename)
            if position is not None:
                filename.seek(position, 0)
            if is_format:
                break
        else:
            raise TypeError('Unknown format for file %s' % filename)
    else:
        # format given via argument
        format = format.upper()
        try:
            format_ep = eps[format]
        except (KeyError, IndexError):
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(eps)))
    # file format should be known by now
    try:
        # search readFormat for given entry point
        read_format = buffered_load_entry_point(
            format_ep.dist.key,
            'obspy.plugin.%s.%s' % (plugin_type, format_ep.name),
            'readFormat')
    except ImportError:
        msg = "Format \"%s\" is not supported. Supported types: %s"
        raise TypeError(msg % (format_ep.name, ', '.join(eps)))
    # read
    list_obj = read_format(filename, **kwargs)
    return list_obj, format_ep.name


def get_script_dir_name():
    """
    Get the directory of the current script file. This is more robust than
    using __file__.
    """
    return str(Path(inspect.getfile(
        inspect.currentframe())).parent.resolve())


def make_format_plugin_table(group="waveform", method="read", numspaces=4,
                             unindent_first_line=True):
    """
    Returns a markdown formatted table with read waveform plugins to insert
    in docstrings.

    >>> table = make_format_plugin_table("event", "write", 4, True)
    >>> print(table)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ======... ===========... ========================================...
    Format    Used Module    _`Linked Function Call`
    ======... ===========... ========================================...
    CMTSOLUTION  :mod:`...io.cmtsolution` :func:`..._write_cmtsolution`
    CNV       :mod:`...io.cnv`   :func:`obspy.io.cnv.core._write_cnv`
    HYPODDPHA :mod:`...io.hypodd`    :func:`obspy.io.hypodd.pha._write_pha`
    JSON      :mod:`...io.json`  :func:`obspy.io.json.core._write_json`
    KML       :mod:`obspy.io.kml` :func:`obspy.io.kml.core._write_kml`
    NLLOC_OBS :mod:`...io.nlloc` :func:`obspy.io.nlloc.core.write_nlloc_obs`
    NORDIC    :mod:`obspy.io.nordic` :func:`obspy.io.nordic.core.write_select`
    QUAKEML :mod:`...io.quakeml` :func:`obspy.io.quakeml.core._write_quakeml`
    SC3ML   :mod:`...io.seiscomp` :func:`obspy.io.seiscomp.event._write_sc3ml`
    SCARDEC   :mod:`obspy.io.scardec`
                             :func:`obspy.io.scardec.core._write_scardec`
    SHAPEFILE :mod:`obspy.io.shapefile`
                             :func:`obspy.io.shapefile.core._write_shapefile`
    ZMAP      :mod:`...io.zmap`  :func:`obspy.io.zmap.core._write_zmap`
    ======... ===========... ========================================...

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

    method = "%sFormat" % method
    eps = _get_ordered_entry_points("obspy.plugin.%s" % group, method,
                                    WAVEFORM_PREFERRED_ORDER)
    mod_list = []
    for name, ep in eps.items():
        module_short = ":mod:`%s`" % ".".join(ep.module_name.split(".")[:3])
        ep_list = [ep.dist.key, "obspy.plugin.%s.%s" % (group, name), method]
        entry_info = str(get_entry_info(*ep_list))
        func_str = ':func:`%s`' % entry_info.split(' = ')[1].replace(':', '.')
        mod_list.append((name, module_short, func_str))

    mod_list = sorted(mod_list)
    headers = ["Format", "Used Module", "_`Linked Function Call`"]
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


def _add_format_plugin_table(func, group, method, numspaces=4):
    """
    A function to populate the docstring of func with its plugin table.
    """
    if func.__doc__ is not None and '%s' in func.__doc__:
        func.__doc__ = func.__doc__ % make_format_plugin_table(
            group, method, numspaces=numspaces)


class ComparingObject(object):
    """
    Simple base class that implements == and != based on self.__dict__
    """

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


def _get_deprecated_argument_action(old_name, new_name, real_action='store'):
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


def sanitize_filename(filename):
    """
    Adapted from Django's slugify functions.

    :param filename: The filename.
    """
    try:
        filename = filename.decode()
    except AttributeError:
        pass

    value = unicodedata.normalize('NFKD', filename).encode(
        'ascii', 'ignore').decode('ascii')
    # In constrast to django we allow dots and don't lowercase.
    value = re.sub(r'[^\w\.\s-]', '', value).strip()
    return re.sub(r'[-\s]+', '-', value)


def download_to_file(url, filename_or_buffer, chunk_size=1024):
    """
    Helper function to download a potentially large file.

    :param url: The URL to GET the data from.
    :type url: str
    :param filename_or_buffer: The filename_or_buffer or file-like object to
        download to.
    :type filename_or_buffer: str or file-like object
    :param chunk_size: The chunk size in bytes.
    :type chunk_size: int
    """
    import requests
    # Workaround for old request versions.
    try:
        r = requests.get(url, stream=True)
    except TypeError:
        r = requests.get(url)

    # Raise anything except for 200
    if r.status_code != 200:
        raise requests.HTTPError('%s HTTP Error: %s for url: %s'
                                 % (r.status_code, r.reason, url))

    if hasattr(filename_or_buffer, "write"):
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            filename_or_buffer.write(chunk)
    else:
        with io.open(filename_or_buffer, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)


def _generic_reader(pathname_or_url=None, callback_func=None,
                    **kwargs):
    # convert pathlib.Path objects to str for compatibility.
    if isinstance(pathname_or_url, PurePath):
        pathname_or_url = str(pathname_or_url)
    if not isinstance(pathname_or_url, str):
        # not a string - we assume a file-like object
        try:
            # first try reading directly
            generic = callback_func(pathname_or_url, **kwargs)
        except TypeError:
            # if this fails, create a temporary file which is read directly
            # from the file system
            pathname_or_url.seek(0)
            with NamedTemporaryFile() as fh:
                fh.write(pathname_or_url.read())
                generic = callback_func(fh.name, **kwargs)
        return generic
    elif isinstance(pathname_or_url, bytes) and \
            pathname_or_url.strip().startswith(b'<'):
        # XML string
        return callback_func(io.BytesIO(pathname_or_url), **kwargs)
    elif "://" in pathname_or_url[:10]:
        # URL
        # extract extension if any
        suffix = Path(Path(pathname_or_url).name).suffix
        if suffix == '':
            suffix = ".tmp"
        with NamedTemporaryFile(suffix=sanitize_filename(suffix)) as fh:
            download_to_file(url=pathname_or_url, filename_or_buffer=fh)
            generic = callback_func(fh.name, **kwargs)
        return generic
    else:
        pathname = pathname_or_url
        # File name(s)
        pathnames = sorted(glob.glob(pathname))
        if not pathnames:
            # try to give more specific information why the stream is empty
            if glob.has_magic(pathname) and not glob.glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not glob.has_magic(pathname) and not Path(pathname).is_file():
                raise IOError(2, "No such file or directory", pathname)

        generic = callback_func(pathnames[0], **kwargs)
        if len(pathnames) > 1:
            for filename in pathnames[1:]:
                generic.extend(callback_func(filename, **kwargs))
        return generic


class CatchAndAssertWarnings(warnings.catch_warnings):
    def __init__(self, clear=None, expected=None, show_all=True, **kwargs):
        """
        :type clear: list[str]
        :param clear: list of modules to clear warning
            registries on (e.g. ``["obspy.signal", "obspy.core"]``), in order
            to make sure any expected warnings will be shown and not suppressed
            because already raised in previously executed code.
        :type expected: list
        :param expected: list of 2-tuples specifying expected
            warnings that should be looked for when exiting the context
            manager. An ``AssertionError`` will be raised if any expected
            warning is not encountered. First item in tuple should be the
            class of the warning, second item should be a regex matching (a
            part of) the warning message (e.g.
            ``(ObsPyDeprecationWarning, 'Attribute .* is deprecated')``).
            Make sure to escape regex special characters like `(` or `.` with a
            backslash and provide message regex as a raw string.
        :type show_all: str
        :param show_all: Whether to set ``warnings.simplefilter('always')``
            when entering context.
        """
        self.registries_to_clear = clear
        self.expected_warnings = expected
        self.show_all = show_all
        # always record warnings, obviously..
        kwargs['record'] = True
        super(CatchAndAssertWarnings, self).__init__(**kwargs)

    def __enter__(self):
        self.warnings = super(CatchAndAssertWarnings, self).__enter__()
        if self.registries_to_clear:
            for modulename in self.registries_to_clear:
                self.clear_warning_registry(modulename)
        if self.show_all:
            warnings.simplefilter("always", Warning)
        # this will always return the list of warnings because we set
        # record=True
        return self.warnings

    def __exit__(self, *exc_info):
        super(CatchAndAssertWarnings, self).__exit__(self, *exc_info)
        # after cleanup, check expected warnings
        self._assert_warnings()

    @staticmethod
    def clear_warning_registry(modulename):
        """
        Clear warning registry of specified module

        :type modulename: str
        :param modulename: Full module name (e.g. ``'obspy.signal'``)
        """
        mod = importlib.import_module(modulename)
        try:
            registry = mod.__warningregistry__
        except AttributeError:
            pass
        else:
            registry.clear()

    def _assert_warnings(self):
        """
        Checks for expected warnings and raises an AssertionError if anyone of
        these is not encountered.
        """
        if not self.expected_warnings:
            return
        for category, regex in self.expected_warnings:
            for warning in self.warnings:
                if not isinstance(warning.message, category):
                    continue
                if not re.search(regex, str(warning.message)):
                    continue
                # found a matching warning, so break out
                break
            else:
                msg = 'Expected warning not raised: (%s, %s)'
                raise AssertionError(msg % (category.__name__, regex))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
