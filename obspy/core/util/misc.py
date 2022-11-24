# -*- coding: utf-8 -*-
"""
Various additional utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import contextlib
import inspect
import io
import itertools
import math
import os
import shutil
import sys
import tempfile
import warnings
from subprocess import STDOUT, CalledProcessError, check_output


import numpy as np
from pkg_resources import load_entry_point

WIN32 = sys.platform.startswith('win32')

# The following dictionary maps the first character of the channel_id to the
# lowest sampling rate this so called Band Code should be used for according
# to SEED MANUAL p.124
# It can e.g. be used to request additional samples on both start and end to
# cut to the samples that really are nearest to requested start/end time
# afterwards.
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
             'Q': 0.00000001
             }

# Dict that stores results from load entry points
_ENTRY_POINT_CACHE = {}

# The kwargs used by load_entry_point function
_LOAD_ENTRY_POINT_KEYS = ('dist', 'group', 'name')


def guess_delta(channel):
    """
    Estimate time delta in seconds between each sample from given channel name.

    :type channel: str
    :param channel: Channel name, e.g. ``'BHZ'`` or ``'H'``
    :rtype: float
    :return: Returns ``0`` if band code is not given or unknown.

    .. rubric:: Example

    >>> print(guess_delta('BHZ'))
    0.1

    >>> print(guess_delta('H'))
    0.0125

    >>> print(guess_delta('XZY'))  # doctest: +SKIP
    0
    """
    try:
        return 1. / BAND_CODE[channel[0]]
    except Exception:
        msg = "No or unknown channel id provided. Specifying a channel id " + \
              "could lead to better selection of first/last samples of " + \
              "fetched traces."
        warnings.warn(msg)
    return 0


def score_at_percentile(values, per, limit=(), issorted=True):
    """
    Calculates the score at the given per percentile of the sequence a.

    For example, the score at ``per=50`` is the median.

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter ``limit`` is provided, it should be a tuple (lower,
    upper) of two values.  Values of ``a`` outside this (closed) interval
    will be ignored.

    .. rubric:: Examples

    >>> a = [1, 2, 3, 4]
    >>> score_at_percentile(a, 25)
    1.75
    >>> score_at_percentile(a, 50)
    2.5
    >>> score_at_percentile(a, 75)
    3.25

    >>> a = [6, 47, 49, 15, 42, 41, 7, 255, 39, 43, 40, 36, 500]
    >>> score_at_percentile(a, 25, limit=(0, 100))
    25.5
    >>> score_at_percentile(a, 50, limit=(0, 100))
    40
    >>> score_at_percentile(a, 75, limit=(0, 100))
    42.5

    This function is taken from `scipy.stats.score_at_percentile`.

    Copyright (c) Gary Strangman
    """
    if limit:
        values = [v for v in values if limit[0] < v < limit[1]]

    if issorted:
        values = sorted(values)

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


def flat_not_masked_contiguous(a):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    This function is taken from
    :func:`numpy.ma.flatnotmasked_contiguous`.

    Copyright (c) Pierre Gerard-Marchant
    """
    np.ma.extras.flatnotmasked_contiguous
    m = np.ma.getmask(a)
    if m is np.ma.nomask:
        return slice(0, a.size, None)
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result


def complexify_string(line):
    """
    Converts a string in the form "(real, imag)" into a complex type.

    :type line: str
    :param line: String in the form ``"(real, imag)"``.
    :rtype: complex
    :return: Complex number.

    .. rubric:: Example

    >>> complexify_string("(1,2)")
    (1+2j)

    >>> complexify_string(" ( 1 , 2 ) ")
    (1+2j)
    """
    temp = line.split(',')
    return complex(float(temp[0].strip()[1:]), float(temp[1].strip()[:-1]))


def to_int_or_zero(value):
    """
    Converts given value to an integer or returns 0 if it fails.

    :param value: Arbitrary data type.
    :rtype: int

    .. rubric:: Example

    >>> to_int_or_zero("12")
    12

    >>> to_int_or_zero("x")
    0
    """
    try:
        return int(value)
    except ValueError:
        return 0


def get_untracked_files_from_git():
    """
    Tries to return a list of files (absolute paths) that are untracked by git
    in the repository.

    Returns `None` if the system call to git fails.
    """
    dir_ = os.path.abspath(
        os.path.dirname(inspect.getfile(inspect.currentframe())))
    dir_ = os.path.dirname(os.path.dirname(os.path.dirname(dir_)))
    try:
        # Check that the git root directory is actually the ObsPy directory.
        p = check_output(['git', 'rev-parse', '--show-toplevel'],
                         cwd=dir_, stderr=STDOUT)
        git_root_dir = p.decode().strip()
        if git_root_dir:
            git_root_dir = os.path.abspath(git_root_dir)
        if os.path.normcase(git_root_dir) != os.path.normcase(dir_):
            raise ValueError('Git root directory (%s) does not match expected '
                             'path (%s).' % (git_root_dir, dir_))
        p = check_output(['git', 'status', '-u', '--porcelain'],
                         cwd=dir_, stderr=STDOUT)
        stdout = p.decode().splitlines()
        files = [os.path.abspath(os.path.join(dir_, line.split()[1].strip()))
                 for line in stdout
                 if line.startswith("??")]
    except (OSError, CalledProcessError):
        return None
    return files


class CaptureIO(io.TextIOWrapper):
    def __init__(self):
        super(CaptureIO, self).__init__(io.BytesIO(), encoding='utf-8',
                                        newline='\n', write_through=True)

    def getvalue(self):
        return self.buffer.getvalue().decode('utf-8')


@contextlib.contextmanager
def CatchOutput():  # NOQA
    """
    A context manager that captures input to stdout/stderr. Python level only!

    Always use with "with" statement. Does nothing otherwise.

    >>> with CatchOutput() as out:  # doctest: +SKIP
    ...    sys.stdout.write("mystdout")
    ...    sys.stderr.write("mystderr")
    >>> print(out.stdout)  # doctest: +SKIP
    mystdout
    >>> print(out.stderr)  # doctest: +SKIP
    mystderr
    """
    # Dummy class to transport the output.
    class Output():
        stdout = ''
        stderr = ''
    out = Output()

    # set current stdout/stderr to in-memory text streams
    sys.stdout = stdout_result = CaptureIO()
    sys.stderr = stderr_result = CaptureIO()

    try:
        raised = False
        yield out
    except SystemExit:
        raised = True
    finally:
        out.stdout = stdout_result.getvalue()
        out.stderr = stderr_result.getvalue()

        # reset to original stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # normalize line breaks
        if WIN32:
            out.stdout = out.stdout.replace('\r', '')
            out.stderr = out.stderr.replace('\r', '')

    if raised:
        raise SystemExit(out.stderr)


def _py3_windowsconsoleio_workaround():
    """
    This monkey patch prevents crashing Py >=3.6 under Windows while using
    the SuppressOutput context manager.

    Python 3.6 implemented unicode console handling for Windows. This works
    by reading/writing to the raw console handle using
    ``{Read,Write}ConsoleW``.
    The problem is that we are going to ``dup2`` over the stdio file
    descriptors when doing ``FDCapture`` and this will ``CloseHandle`` the
    handles used by Python to write to the console. Though there is still some
    weirdness and the console handle seems to only be closed randomly and not
    on the first call to ``CloseHandle``, or maybe it gets reopened with the
    same handle value when we suspend capturing.
    The workaround in this case will reopen stdio with a different fd which
    also means a different handle by replicating the logic in
    "Py_lifecycle.c:initstdio/create_stdio".

    See https://github.com/pytest-dev/py/issues/103
    See http://bugs.python.org/issue30555
    See https://github.com/obspy/obspy/issues/3148#issuecomment-1274254892
    """
    if not WIN32 or sys.version_info[:2] < (3, 6):
        return
    if not hasattr(sys.stdout, 'buffer'):
        return
    buffered = hasattr(sys.stdout.buffer, 'raw')
    raw_stdout = sys.stdout.buffer.raw if buffered else sys.stdout.buffer

    if not isinstance(raw_stdout, io._WindowsConsoleIO):
        return

    def _reopen_stdio(f, mode):
        if not buffered and mode[0] == 'w':
            buffering = 0
        else:
            buffering = -1

        return io.TextIOWrapper(
            open(os.dup(f.fileno()), mode, buffering),
            f.encoding,
            f.errors,
            f.newlines,
            f.line_buffering)

    sys.stdin = _reopen_stdio(sys.stdin, 'rb')
    sys.stdout = _reopen_stdio(sys.stdout, 'wb')
    sys.stderr = _reopen_stdio(sys.stderr, 'wb')


@contextlib.contextmanager
def SuppressOutput():  # noqa
    """
    A context manager that suppresses output to stdout/stderr.
    Always use with "with" statement. Does nothing otherwise.
    >>> with SuppressOutput():  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')

    Note: Does not work reliably for Python 3 under Windows - see
    function definition of _py3_windowsconsoleio_workaround().
    """
    _py3_windowsconsoleio_workaround()
    with os.fdopen(os.dup(1), 'wb', 0) as tmp_stdout:
        with os.fdopen(os.dup(2), 'wb', 0) as tmp_stderr:
            with open(os.devnull, 'wb') as to_file:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(to_file.fileno(), 1)
                os.dup2(to_file.fileno(), 2)
                try:
                    yield
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(tmp_stdout.fileno(), 1)
                    os.dup2(tmp_stderr.fileno(), 2)
    # reset to original stdout/stderr
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


@contextlib.contextmanager
def TemporaryWorkingDirectory():  # noqa --> this name is IMHO ok for a CM
    """
    A context manager that changes to a temporary working directory.

    Always use with "with" statement. Does nothing useful otherwise.

    >>> with TemporaryWorkingDirectory():  # doctest: +SKIP
    ...    os.system('echo "$PWD"')
    """
    tempdir = tempfile.mkdtemp(prefix='obspy-')
    old_dir = os.getcwd()
    os.chdir(tempdir)
    try:
        yield tempdir
    finally:
        os.chdir(old_dir)
        try:
            shutil.rmtree(tempdir)
        # Windows on appveyor is having issues with removing these temporary
        # directories (e.g. https://ci.appveyor.com/project/obspy/obspy/builds
        # /20718605/job/0vuja1b95rv5a0s2).
        except Exception as e:
            warnings.warn(e.__repr__())


def factorize_int(x):
    """
    Calculate prime factorization of integer.

    Could be done faster but faster algorithm have much more lines of code and
    this is fast enough for our purposes.

    https://stackoverflow.com/q/14550794

    >>> factorize_int(1800004)
    [2, 2, 450001]
    >>> factorize_int(1800003)
    [3, 19, 23, 1373]
    """
    if x == 1:
        return [1]
    factors, limit, check, num = [], int(math.sqrt(x)) + 1, 2, x
    for check in range(2, limit):
        while num % check == 0:
            factors.append(check)
            num /= check
    if num > 1:
        factors.append(int(num))
    return factors


def get_window_times(starttime, endtime, window_length, step, offset,
                     include_partial_windows):
    """
    Function calculating a list of times making up equal length windows from
    within a given time interval.

    :param starttime: The start time of the whole time interval.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: The end time of the whole time interval.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param window_length: The length of each window in seconds.
    :type window_length: float
    :param step: The step between the start times of two successive
        windows in seconds. Can be negative if an offset is given.
    :type step: float
    :param offset: The offset of the first window in seconds relative to
        the start time of the whole interval.
    :type offset: float
    :param include_partial_windows: Determines if windows that are
        shorter then 99.9 % of the desired length are returned.
    :type include_partial_windows: bool
    """
    if step > 0:
        end = endtime.timestamp - 0.001 * step
    else:
        # This minus is correct here due to the way the actual window times
        # are calculate later on.
        end = starttime.timestamp - 0.001 * abs(step)
    # Left sides of each window.
    indices = np.arange(start=starttime.timestamp + offset,
                        stop=end, step=step, dtype=np.float64)
    if step > 0:
        # Generate all possible windows.
        windows = [(_i, min(_i + window_length, endtime.timestamp))
                   for _i in indices]
    else:
        # Generate all possible windows.
        windows = [(max(_i - window_length, starttime.timestamp), _i)
                   for _i in indices]

    # Potentially remove partial windows not fulfilling the window length
    # criterion.
    if not include_partial_windows:
        windows = [_i for _i in windows
                   if abs(_i[1] - _i[0]) > 0.999 * window_length]

    t = type(starttime)
    return [(t(_i[0]), t(_i[1])) for _i in windows]


class MatplotlibBackend(object):
    """
    A helper class for switching the matplotlib backend.

    Can be used as a context manager to temporarily switch the backend or by
    using the :meth:`~MatplotlibBackend.switch_backend` staticmethod.

    The context manager has no effect when setting ``backend=None``.

    :type backend: str
    :param backend: Name of matplotlib backend to switch to.
    :type sloppy: bool
    :param sloppy: If ``True``, uses :func:`matplotlib.pyplot.switch_backend`
        and no warning will be shown if the backend was not switched
        successfully. If ``False``, additionally tries to use
        :func:`matplotlib.use` first and also shows a warning if the backend
        was not switched successfully.
    :type close: bool
    :param close: Whether to close all matplotlib figures when exiting the
        context manager.
    """

    def __init__(self, backend, sloppy=True, close=False):
        self.temporary_backend = backend
        self.sloppy = sloppy
        import matplotlib
        self.previous_backend = matplotlib.get_backend()
        self.close = close

    def __enter__(self):
        if self.temporary_backend is None:
            return
        self.switch_backend(backend=self.temporary_backend, sloppy=self.sloppy)

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        if self.close:
            import matplotlib.pyplot as plt
            plt.close('all')
        if self.temporary_backend is None:
            return
        self.switch_backend(backend=self.previous_backend, sloppy=self.sloppy)

    @staticmethod
    def switch_backend(backend, sloppy=True):
        """
        Switch matplotlib backend.

        :type backend: str
        :param backend: Name of matplotlib backend to switch to.
        :type sloppy: bool
        :param sloppy: If ``True``, only uses
            :func:`matplotlib.pyplot.switch_backend` and no warning will be
            shown if the backend was not switched successfully. If ``False``,
            additionally tries to use :func:`matplotlib.use` first and also
            shows a warning if the backend was not switched successfully.
        """
        import matplotlib
        # sloppy. only do a `plt.switch_backend(..)`
        if sloppy:
            import matplotlib.pyplot as plt
            plt.switch_backend(backend)
        else:
            # check if `matplotlib.use(..)` is emitting a warning
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("error", UserWarning)
                    matplotlib.use(backend)
            # if that's the case, follow up with `plt.switch_backend(..)`
            except UserWarning:
                import matplotlib.pyplot as plt
                plt.switch_backend(backend)
            # finally check if the switch was successful,
            # show a warning if not
            if matplotlib.get_backend().upper() != backend.upper():
                msg = "Unable to change matplotlib backend to '%s'" % backend
                raise Exception(msg)


def limit_numpy_fft_cache(max_size_in_mb_per_cache=100):
    """
    NumPy's FFT implementation utilizes caches to speedup subsequent FFTs of
    the same size. This accumulates memory when run for various length FFTs
    as can readily happen in seismology.

    This utility function clears both, full and real-only caches if their
    size is above the given threshold.

    The default 100 MB is fairly generous but we still want to profit from
    the cache where applicable.
    """
    for cache in ["_fft_cache", "_real_fft_cache"]:
        # Guard against different numpy versions just to be safe.
        if not hasattr(np.fft.fftpack, cache):
            continue
        cache = getattr(np.fft.fftpack, cache)
        # Check type directly and don't use isinstance() as future numpy
        # versions might use some subclass or what not.
        if type(cache) is not dict:
            continue
        # Its a dictionary with list's of arrays as the values. Wrap in
        # try/except to guard against future numpy changes.
        try:
            total_size = sum([_j.nbytes for _i in cache.values() for _j in _i])
        except Exception:
            continue
        if total_size > max_size_in_mb_per_cache * 1024 * 1024:
            cache.clear()


def buffered_load_entry_point(dist, group, name):
    """
    Return `name` entry point of `group` for `dist` or raise ImportError
    :type dist: str
    :param dist: The name of the distribution containing the entry point.
    :type group: str
    :param group: The name of the group containing the entry point.
    :type name: str
    :param name: The name of the entry point.
    :return: The loaded entry point
    """
    hash_str = '/'.join([dist, group, name])
    if hash_str not in _ENTRY_POINT_CACHE:
        _ENTRY_POINT_CACHE[hash_str] = load_entry_point(dist, group, name)
    return _ENTRY_POINT_CACHE[hash_str]


def _yield_obj_parent_attr(obj, cls=None, is_attr=None, has_attr=None):
    """
    Recurse an object, yield a tuple of object, parent, attr.

    Can be used, for example, to yield all ResourceIdentifier instances
    contained in any obspy.core.event class instances and attached instances,
    as well as the objects they are attached to (parents) and the attribute
    name in which they are stored (attr).

    :param obj:
        The object to recurse through attributes of lists, tuples, and other
        instances.
    :param cls:
        Only return instances of cls if not None, else return all instances.
    :param is_attr:
        Only return objects stored as attr_name, if None return all.
    :param has_attr:
        Only return objects that have attribute has_attr, if None return all.

    .. rubric:: General Usage

    Get a list of all resource_ids contained in an event, the objects they
    are attached to, and the attribute name on the parent object.

    >>> import obspy
    >>> from obspy.core.event import ResourceIdentifier
    >>> cat = obspy.read_events()
    >>> resource_tuple = list(_yield_obj_parent_attr(cat, ResourceIdentifier))
    """
    ids = set()  # id cache to avoid circular references

    def func(obj, attr=None, parent=None):
        id_tuple = (id(obj), id(parent))

        # If object/parent combo have not been yielded continue.
        if id_tuple not in ids:
            ids.add(id_tuple)
            # Check if this object is stored as the desired attribute.
            is_attribute = is_attr is None or attr == is_attr
            # Check if the object has the desired attribute.
            has_attribute = has_attr is None or hasattr(obj, has_attr)
            # Check if isinstance of desired class.
            is_instance = cls is None or isinstance(obj, cls)
            # Yield object, parent, and attr if desired conditions are met.
            if is_attribute and has_attribute and is_instance:
                yield (obj, parent, attr)
            # Iterate through basic built-in types.
            if isinstance(obj, (list, tuple)):
                for val in obj:
                    for out in func(val, attr=attr, parent=obj):
                        yield out
            elif isinstance(obj, dict):
                for item, val in obj.items():
                    for out in func(val, attr=item, parent=obj):
                        yield out
            # Iterate through non built-in object attributes.
            elif hasattr(obj, '__slots__'):
                for attr in obj.__slots__:
                    val = getattr(obj, attr)
                    for out in func(val, attr=attr, parent=obj):
                        yield out
            elif hasattr(obj, '__dict__'):
                for item, val in obj.__dict__.items():
                    for out in func(val, attr=item, parent=obj):
                        yield out

    return func(obj)


def _yield_resource_id_parent_attr(obj):
    """
    Specialized form of _yield_obj_parent_attr for getting ResourceIdentifiers.

    This function makes some assumptions because only resource_identifiers are
    being sought in order to improve efficiency.
    """
    from obspy.core.event import ResourceIdentifier

    ids = set()  # id cache to avoid circular references

    def func(obj, parent=None, attr=None):
        if obj is None or not (hasattr(obj, '__dict__') or
                               isinstance(obj, (list, tuple))):
            return  # stop iteration
        id_tuple = (id(obj), id(parent))
        if id_tuple not in ids:
            ids.add(id_tuple)
            # Yield object, parent, and attr if desired conditions are met
            if isinstance(obj, ResourceIdentifier):
                yield (obj, parent, attr)
            # Iterate through basic built-in types.
            elif isinstance(obj, (list, tuple)):
                for val in obj:
                    for out in func(val, obj, attr):
                        yield out
            # Iterate through non built-in object attributes.
            elif hasattr(obj, '__dict__'):
                for item, val in obj.__dict__.items():
                    for out in func(val, obj, item):
                        yield out

    return func(obj)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
