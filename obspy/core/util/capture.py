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
from future.utils import PY2

import io
import os
import platform
import sys
import tempfile
from contextlib import contextmanager


if PY2:
    CaptureIO = io.StringIO
else:
    class CaptureIO(io.TextIOWrapper):
        def __init__(self):
            super(CaptureIO, self).__init__(
                io.BytesIO(),
                encoding='UTF-8', newline='', write_through=True,
            )

        def getvalue(self):
            return self.buffer.getvalue()


@contextmanager
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


@contextmanager
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
                sys.stdout.flush()
                os.dup2(tmp_stdout.fileno(), stdout_fd)

                sys.stderr.flush()
                os.dup2(tmp_stderr.fileno(), stderr_fd)

                raised = False
                yield out

            except SystemExit:
                raised = True

            finally:
                sys.stdout.flush()
                os.dup2(stdout_copy, stdout_fd)
                os.close(stdout_copy)
                tmp_stdout.seek(0)
                out.stdout = tmp_stdout.read()

                sys.stderr.flush()
                os.dup2(stderr_copy, stderr_fd)
                os.close(stderr_copy)
                tmp_stderr.seek(0)
                out.stderr = tmp_stderr.read()

                if platform.system() == "Windows":
                    out.stdout = out.stdout.replace(b'\r', b'')
                    out.stderr = out.stderr.replace(b'\r', b'')

                if raised:
                    raise SystemExit(out.stderr)
