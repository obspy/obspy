# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2

import os
import sys
import tempfile
import unittest
import warnings

from obspy import UTCDateTime, read
from obspy.core.compatibility import mock
from obspy.core.event import ResourceIdentifier as ResId
from obspy.core.util.misc import CatchOutput, get_window_times, \
    _ENTRY_POINT_CACHE, _yield_obj_parent_attr
from obspy.core.util.testing import WarningsCapture


class UtilMiscTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.misc
    """

    def test_supress_output(self):
        """
        Tests that CatchOutput suppresses messages
        """
        # this should write nothing to console
        with CatchOutput():
            sys.stdout.write("test_suppress_output #1 failed")
            sys.stderr.write("test_suppress_output #2 failed")
            print("test_suppress_output #3 failed")
            print("test_suppress_output #4 failed", file=sys.stdout)
            print("test_suppress_output #5 failed", file=sys.stderr)

    def test_catch_output(self):
        """
        Tests that CatchOutput catches messages
        """
        with CatchOutput() as out:
            sys.stdout.write("test_catch_output #1")
            sys.stderr.write("test_catch_output #2")
        self.assertEqual(out.stdout, 'test_catch_output #1')
        self.assertEqual(out.stderr, 'test_catch_output #2')

        with CatchOutput() as out:
            print("test_catch_output #3")
        self.assertEqual(out.stdout, 'test_catch_output #3\n')
        self.assertEqual(out.stderr, '')

        with CatchOutput() as out:
            print("test_catch_output #4", file=sys.stdout)
            print("test_catch_output #5", file=sys.stderr)
        self.assertEqual(out.stdout, 'test_catch_output #4\n')
        self.assertEqual(out.stderr, 'test_catch_output #5\n')

    def test_catch_output_bytes(self):
        with CatchOutput() as out:
            if PY2:
                sys.stdout.write(b"test_catch_output_bytes #1")
                sys.stderr.write(b"test_catch_output_bytes #2")
            else:
                # PY3 does not allow to write directly bytes into text streams
                sys.stdout.buffer.write(b"test_catch_output_bytes #1")
                sys.stderr.buffer.write(b"test_catch_output_bytes #2")
        self.assertEqual(out.stdout, 'test_catch_output_bytes #1')
        self.assertEqual(out.stderr, 'test_catch_output_bytes #2')

    def test_catch_output_io(self):
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
        files = ["libnames.py", "version.py"]

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

    def test_get_window_times(self):
        """
        Tests for the get_window_times() helper function.
        """
        # Basic windows. 4 pieces.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=5.0,
                step=5.0,
                offset=0.0,
                include_partial_windows=False),
            [
                (UTCDateTime(0), UTCDateTime(5)),
                (UTCDateTime(5), UTCDateTime(10)),
                (UTCDateTime(10), UTCDateTime(15)),
                (UTCDateTime(15), UTCDateTime(20))
            ]
        )

        # Different step size.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=5.0,
                step=10.0,
                offset=0.0,
                include_partial_windows=False),
            [
                (UTCDateTime(0), UTCDateTime(5)),
                (UTCDateTime(10), UTCDateTime(15))
            ]
        )

        # With offset.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=5.0,
                step=6.5,
                offset=8.5,
                include_partial_windows=False),
            [
                (UTCDateTime(8.5), UTCDateTime(13.5)),
                (UTCDateTime(15), UTCDateTime(20))
            ]
        )

        # Don't return partial windows.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=15.0,
                step=15.0,
                offset=0.0,
                include_partial_windows=False),
            [
                (UTCDateTime(0), UTCDateTime(15))
            ]
        )

        # Return partial windows.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=15.0,
                step=15.0,
                offset=0.0,
                include_partial_windows=True),
            [
                (UTCDateTime(0), UTCDateTime(15)),
                (UTCDateTime(15), UTCDateTime(20))
            ]
        )

        # Negative step length has to be used together with an offset.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=5.0,
                step=-5.0,
                offset=20.0,
                include_partial_windows=False),
            [
                (UTCDateTime(15), UTCDateTime(20)),
                (UTCDateTime(10), UTCDateTime(15)),
                (UTCDateTime(5), UTCDateTime(10)),
                (UTCDateTime(0), UTCDateTime(5))
            ]
        )

        # Negative step length and not partial windows.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=15.0,
                step=-15.0,
                offset=20.0,
                include_partial_windows=False),
            [
                (UTCDateTime(5), UTCDateTime(20))
            ]
        )

        # Negative step length with partial windows.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(20),
                window_length=15.0,
                step=-15.0,
                offset=20.0,
                include_partial_windows=True),
            [
                (UTCDateTime(5), UTCDateTime(20)),
                (UTCDateTime(0), UTCDateTime(5))
            ]
        )

        # Smaller step than window.
        self.assertEqual(
            get_window_times(
                starttime=UTCDateTime(0),
                endtime=UTCDateTime(2),
                window_length=1.0,
                step=0.25,
                offset=0.0,
                include_partial_windows=False),
            [
                (UTCDateTime(0), UTCDateTime(1)),
                (UTCDateTime(0.25), UTCDateTime(1.25)),
                (UTCDateTime(0.5), UTCDateTime(1.5)),
                (UTCDateTime(0.75), UTCDateTime(1.75)),
                (UTCDateTime(1.0), UTCDateTime(2.0))
            ]
        )

    def test_entry_point_buffer(self):
        """
        Ensure the entry point buffer caches results from load_entry_point
        """
        with mock.patch.dict(_ENTRY_POINT_CACHE, clear=True):
            with mock.patch('obspy.core.util.misc.load_entry_point') as p:
                # raises UserWarning: No matching response information found.
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore', UserWarning)
                    st = read()
                    st.write('temp.mseed', 'mseed')
            self.assertEqual(len(_ENTRY_POINT_CACHE), 3)
            self.assertEqual(p.call_count, 3)

    def test_yield_obj_parent_attr(self):
        """
        Setup a complex data structure and ensure recursive search function
        finds all target objects.
        """
        class Slots(object):
            """
            A simple class with slots
            """
            __slots__ = ('right', )

            def __init__(self, init):
                self.right = init

        slotted = Slots((ResId('1'), AttributeError, [ResId('2')]))
        nested = {
            'not_right': 'nope',
            'good': {'right': ResId('3'), 'wrong': [1, [(())]]},
            'right': [[[[[[[[ResId('4')]]], ResId('5')]]]]],
        }

        base = dict(right=ResId('6'), slotted=slotted, nested=nested)

        out = list(_yield_obj_parent_attr(base, ResId))

        self.assertEqual(len(out), 6)

        for obj, parent, attr in out:
            self.assertEqual(attr, 'right')
            self.assertIsInstance(obj, ResId)

    def test_warning_capture(self):
        """
        Tests for the WarningsCapture class in obspy.core.util.testing
        """
        # ensure a warning issued with warn is captured. Before, this could
        # raise a TypeError.
        with WarningsCapture() as w:
            warnings.warn('something bad is happening in the world')

        self.assertEqual(len(w), 1)
        self.assertIn('something bad', str(w.captured_warnings[0].message))


def suite():
    return unittest.makeSuite(UtilMiscTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
