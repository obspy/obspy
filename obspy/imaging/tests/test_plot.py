# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.plot / obspy-plot test suite.
"""
import shutil
from pathlib import Path

import pytest

from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.imaging.scripts.plot import main as obspy_plot


class TestPlot:
    """
    Test cases for obspy-plot
    """
    @pytest.fixture(scope='class')
    def all_files(self, root):
        """Collect all files. """
        all_files = [root / 'io' / 'ascii' / 'tests' / 'data' / i
                     for i in ['slist.ascii', 'slist_2_traces.ascii']]
        return all_files

    def test_plot(self, image_path, all_files):
        """
        Run obspy-plot on selected tests
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            all_files_list = []
            for filename in all_files:
                newname = Path('.') / filename.name
                shutil.copy(filename, newname)
                all_files_list.append(str(newname))

            obspy_plot(['--outfile', str(image_path)] + all_files_list)

    def test_plot_no_merge(self, image_path, all_files):
        """
        Run obspy-plot without trace merging
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            all_files_list = []
            for filename in all_files:
                newname = Path('.') / filename.name
                shutil.copy(filename, newname)
                all_files_list.append(str(newname))

            obspy_plot(['--no-automerge', '--outfile', str(image_path)] +
                       all_files_list)
