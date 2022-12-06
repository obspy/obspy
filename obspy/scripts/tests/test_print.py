#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

from obspy.scripts.print import main as obspy_print
from obspy.core.util.misc import CatchOutput


class PrintTestCase(unittest.TestCase):
    def setUp(self):
        self.all_files = [os.path.join(os.path.dirname(__file__), os.pardir,
                                       os.pardir, 'io', 'ascii', 'tests',
                                       'data', x)
                          for x in ['slist.ascii', 'tspair.ascii']]

    def test_print(self):
        with CatchOutput() as out:
            obspy_print(self.all_files)

        expected = '''1 Trace(s) in Stream:
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
'''  # noqa
        self.assertEqual(expected, out.stdout)

    def test_print_nomerge(self):
        with CatchOutput() as out:
            obspy_print(['--no-merge'] + self.all_files)

        expected = '''2 Trace(s) in Stream:
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
'''  # noqa
        self.assertEqual(expected, out.stdout)
