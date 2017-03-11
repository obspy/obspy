# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import PY2

from contextlib import contextmanager
import io
import sys


try:
    from contextlib import redirect_stdout
except ImportError:
    # Python < 3.4
    @contextmanager
    def redirect_stdout(new_target):
        old_target, sys.stdout = sys.stdout, new_target
        try:
            yield new_target
        finally:
            sys.stdout = old_target


if PY2:
    from StringIO import StringIO  # @UnresolvedImport
    CaptureIO = StringIO
else:
    class CaptureIO(io.TextIOWrapper):
        def __init__(self):
            super(CaptureIO, self).__init__(io.BytesIO(), encoding='UTF-8',
                                            newline='', write_through=True)

        def getvalue(self):
            return self.buffer.getvalue().decode('UTF-8')


def catch_stdout():
    return redirect_stdout(CaptureIO())
