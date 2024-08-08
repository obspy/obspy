#! /usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from obspy.scripts.print import main as obspy_print
from obspy.core.util.misc import CatchOutput


class TestPrint():
    @pytest.fixture(scope='class')
    def all_files(self, root):
        all_files = [str(root / 'io' / 'ascii' / 'tests' / 'data' / name)
                     for name in ('slist.ascii', 'tspair.ascii')]
        return all_files

    def test_print(self, all_files):
        with CatchOutput() as out:
            obspy_print(all_files)

        expected = '''1 Trace(s) in Stream:
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
'''  # noqa
        assert expected == out.stdout

    def test_print_nomerge(self, all_files):
        with CatchOutput() as out:
            obspy_print(['--no-merge'] + all_files)

        expected = '''2 Trace(s) in Stream:
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
XX.TEST..BHZ | 2008-01-15T00:00:00.025000Z - 2008-01-15T00:00:15.875000Z | 40.0 Hz, 635 samples
'''  # noqa
        assert expected == out.stdout
