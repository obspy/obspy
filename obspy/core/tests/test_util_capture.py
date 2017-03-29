# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import ctypes
import os
import platform
import sys
import tempfile
import unittest

from obspy.core.util.capture import PyCatchOutput, CCatchOutput, SuppressOutput


class UtilCaptureTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.capture
    """
    def test_c_catch_output(self):
        """
        Tests for CatchOutput context manager.
        """
        if sys.platform == 'win32':
            libc = ctypes.CDLL('msvcrt')
        else:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))

        with CCatchOutput() as out:
            os.system('echo "abc"')
            libc.printf(b"def\n")
            os.system('echo "123" 1>&2')

        if platform.system() == "Windows":
            self.assertEqual(out.stdout, b'"abc"\ndef\n')
            self.assertEqual(out.stderr, b'"123" \n')
        else:
            self.assertEqual(out.stdout, b"abc\ndef\n")
            self.assertEqual(out.stderr, b"123\n")

        with CCatchOutput() as out:
            print("ghi")
            print("jkl", file=sys.stdout)
            print("456", file=sys.stderr)

        self.assertEqual(out.stdout, b'ghi\njkl\n')
        self.assertEqual(out.stderr, b'456\n')

    def test_py_catch_output(self):
        """
        Tests that PyCatchOutput context manager does not break I/O.
        """
        with PyCatchOutput():
            fn = tempfile.TemporaryFile(prefix='obspy')

        try:
            fn.write(b'abc')
            fn.seek(0)
            fn.read(3)
            fn.close()
        except OSError as e:
            self.fail('PyCatchOutput has broken file I/O!\n' + str(e))

    def test_suppress_output(self):
        """
        Tests for SuppressOutput context manager.
        """
        if sys.platform == 'win32':
            libc = ctypes.CDLL('msvcrt')
        else:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))

        # this should write nothing to console
        with SuppressOutput():
            os.system('echo "test_SuppressOutput #1 failed"')
            libc.printf(b"test_SuppressOutput #2 failed")
            os.system('echo "test_SuppressOutput #3 failed" 1>&2')
            sys.stderr.write("test_SuppressOutput #4 failed")
            sys.stdout.write("test_SuppressOutput #5 failed")
            print("test_SuppressOutput #6 failed")
            print("test_SuppressOutput #7 failed", file=sys.__stdout__)
            print("test_SuppressOutput #8 failed", file=sys.__stderr__)


def suite():
    return unittest.makeSuite(UtilCaptureTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
