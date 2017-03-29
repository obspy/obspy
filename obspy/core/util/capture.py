# -*- coding: utf-8 -*-
"""
Context managers to catch output to stdout/stderr streams.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str

import contextlib
import ctypes
import io
import os
import platform
import sys
import tempfile


if PY2:
    from cStringIO import StringIO
    CaptureIO = StringIO
else:
    class CaptureIO(io.TextIOWrapper):
        def __init__(self):
            super(CaptureIO, self).__init__(io.BytesIO(), encoding='UTF-8',
                                            newline='', write_through=True)

        def getvalue(self):
            return self.buffer.getvalue()


if sys.platform == 'win32':
    libc = ctypes.CDLL(native_str("msvcrt"))
else:
    libc = ctypes.CDLL(ctypes.util.find_library(native_str("c")))


def flush():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except:
        pass
    try:
        libc.fflush(None)
    except:
        pass


@contextlib.contextmanager
def PyCatchOutput():  # noqa
    """
    A context manager that catches stdout/stderr/exit() for its scope.

    Always use with "with" statement. Does nothing otherwise.

    >>> with PyCatchOutput() as out:  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')
    >>> print(out.stdout)  # doctest: +SKIP
    mystdout
    >>> print(out.stderr)  # doctest: +SKIP
    mystderr
    """

    # Dummy class to transport the output.
    class Output():
        pass
    out = Output()
    out.stdout = ''
    out.stderr = ''

    sys.stdout = temp_out = CaptureIO()
    sys.stderr = temp_err = CaptureIO()

    yield out

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    out.stdout = temp_out.getvalue()
    out.stderr = temp_err.getvalue()

    if platform.system() == "Windows":
        out.stdout = out.stdout.replace(b'\r', b'')
        out.stderr = out.stderr.replace(b'\r', b'')


@contextlib.contextmanager
def PyCatchOutput2():  # noqa
    """
    A context manager that catches stdout/stderr/exit() for its scope.

    Always use with "with" statement. Does nothing otherwise.

    >>> with PyCatchOutput() as out:  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')
    >>> print(out.stdout)  # doctest: +SKIP
    mystdout
    >>> print(out.stderr)  # doctest: +SKIP
    mystderr
    """

    # Dummy class to transport the output.
    class Output():
        pass
    out = Output()
    out.stdout = ''
    out.stderr = ''

    sys.stdout = CaptureIO()
    sys.stderr = CaptureIO()

    yield

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


@contextlib.contextmanager
def CCatchOutput():  # noqa
    """
    A context manager that catches stdout/stderr/exit() for its scope.

    Always use with "with" statement. Does nothing otherwise.

    Based on: https://bugs.python.org/msg184312

    >>> with CCatchOutput() as out:  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')
    >>> print(out.stdout)  # doctest: +SKIP
    mystdout
    >>> print(out.stderr)  # doctest: +SKIP
    mystderr
    """

    # Dummy class to transport the output.
    class Output():
        pass
    out = Output()
    out.stdout = ''
    out.stderr = ''

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    with tempfile.TemporaryFile(prefix='obspy-') as tmp_stdout:
        with tempfile.TemporaryFile(prefix='obspy-') as tmp_stderr:
            stdout_copy = os.dup(stdout_fd)
            stderr_copy = os.dup(stderr_fd)

            try:
                flush()
                os.dup2(tmp_stdout.fileno(), stdout_fd)
                os.dup2(tmp_stderr.fileno(), stderr_fd)

                raised = False
                yield out

            except SystemExit:
                raised = True

            finally:
                flush()
                os.dup2(stdout_copy, stdout_fd)
                os.close(stdout_copy)
                tmp_stdout.seek(0)
                out.stdout = tmp_stdout.read()

                os.dup2(stderr_copy, stderr_fd)
                os.close(stderr_copy)
                tmp_stderr.seek(0)
                out.stderr = tmp_stderr.read()

                if platform.system() == "Windows":
                    out.stdout = out.stdout.replace(b'\r', b'')
                    out.stderr = out.stderr.replace(b'\r', b'')

                if raised:
                    raise SystemExit(out.stderr)


@contextlib.contextmanager
def SuppressOutput():  # noqa
    """
    A context manager that suppresses output to stdout/stderr.

    Always use with "with" statement. Does nothing otherwise.

    >>> with SuppressOutput():  # doctest: +SKIP
    ...    os.system('echo "mystdout"')
    ...    os.system('echo "mystderr" >&2')
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(stdout_fd), 'wb') as tmp_stdout:
        with os.fdopen(os.dup(stderr_fd), 'wb') as tmp_stderr:
            with open(os.devnull, 'wb') as to_file:
                flush()
                os.dup2(to_file.fileno(), stdout_fd)
                os.dup2(to_file.fileno(), stderr_fd)
                try:
                    yield
                finally:
                    flush()
                    os.dup2(tmp_stdout.fileno(), stdout_fd)
                    os.dup2(tmp_stderr.fileno(), stderr_fd)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
