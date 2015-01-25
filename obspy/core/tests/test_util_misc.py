# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from ctypes import CDLL
from ctypes.util import find_library
from obspy.core.util.misc import wrap_long_string, CatchOutput
import os
import platform
import sys
import tempfile
import unittest


class UtilMiscTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.misc
    """
    def test_wrap_long_string(self):
        """
        Tests for the wrap_long_string() function.
        """
        string = ("Retrieve an event based on the unique origin "
                  "ID numbers assigned by the IRIS DMC")
        got = wrap_long_string(string, prefix="\t*\t > ", line_length=50)
        expected = ("\t*\t > Retrieve an event based on\n"
                    "\t*\t > the unique origin ID numbers\n"
                    "\t*\t > assigned by the IRIS DMC")
        self.assertEqual(got, expected)
        got = wrap_long_string(string, prefix="\t* ", line_length=70)
        expected = ("\t* Retrieve an event based on the unique origin ID\n"
                    "\t* numbers assigned by the IRIS DMC")
        got = wrap_long_string(string, prefix="\t \t  > ",
                               special_first_prefix="\t*\t", line_length=50)
        expected = ("\t*\tRetrieve an event based on\n"
                    "\t \t  > the unique origin ID numbers\n"
                    "\t \t  > assigned by the IRIS DMC")
        problem_string = ("Retrieve_an_event_based_on_the_unique "
                          "origin ID numbers assigned by the IRIS DMC")
        got = wrap_long_string(problem_string, prefix="\t\t", line_length=40,
                               sloppy=True)
        expected = ("\t\tRetrieve_an_event_based_on_the_unique\n"
                    "\t\torigin ID\n"
                    "\t\tnumbers\n"
                    "\t\tassigned by\n"
                    "\t\tthe IRIS DMC")
        got = wrap_long_string(problem_string, prefix="\t\t", line_length=40)
        expected = ("\t\tRetrieve_an_event_base\\\n"
                    "\t\td_on_the_unique origin\n"
                    "\t\tID numbers assigned by\n"
                    "\t\tthe IRIS DMC")

    def test_CatchOutput(self):
        """
        Tests for CatchOutput context manager.
        """
        libc = CDLL(find_library("c"))

        with CatchOutput() as out:
            os.system('echo "abc"')
            libc.printf(b"def\n")
            # This flush is necessary for Python 3, which uses different
            # buffering modes. Fortunately, in practice, we do not mix Python
            # and C writes to stdout. This can also be fixed by setting the
            # PYTHONUNBUFFERED environment variable, but this must be done
            # externally, and cannot be done by the script.
            libc.fflush(None)
            print("ghi")
            print("jkl", file=sys.stdout)
            os.system('echo "123" 1>&2')
            print("456", file=sys.stderr)

        if platform.system() == "Windows":
            self.assertEqual(out.stdout, b'"abc"\ndef\nghi\njkl\n')
            self.assertEqual(out.stderr, b'"123" \n456\n')
        else:
            self.assertEqual(out.stdout, b"abc\ndef\nghi\njkl\n")
            self.assertEqual(out.stderr, b"123\n456\n")

    def test_CatchOutput_IO(self):
        """
        Tests that CatchOutput context manager does not break I/O.
        """
        with CatchOutput():
            fn = tempfile.TemporaryFile(prefix='obspy')

        try:
            fn.write(b'abc')
            fn.seek(0)
            fn.read(3)
            fn.close()
        except OSError as e:
            self.fail('CatchOutput has broken file I/O!\n' + str(e))

    def test_no_obspy_imports(self):
        """
        Check files that are used at install time for obspy imports.
        """
        from obspy.core import util
        files = ["misc.py", "version.py"]

        for file_ in files:
            file_ = os.path.join(os.path.dirname(util.__file__), file_)
            msg = ("File %s seems to contain an import 'from obspy' "
                   "(line %%i: '%%s').") % file_
            with open(file_, "rb") as fh:
                lines = fh.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith(b"#"):
                    continue
                if b"from obspy" in line:
                    if b" import " in line:
                        self.fail(msg % (i, line))
                if b"import obspy" in line:
                    self.fail(msg % (i, line))


def suite():
    return unittest.makeSuite(UtilMiscTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
