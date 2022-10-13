# -*- coding: utf-8 -*-
"""
The obspy.imaging.scripts.scan / obspy-scan test suite.
"""
import os
import shutil
import warnings
from os.path import abspath, dirname, join, pardir
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from obspy import read, UTCDateTime
from obspy.core.util.base import NamedTemporaryFile, get_example_file
from obspy.core.util.misc import TemporaryWorkingDirectory, CatchOutput
from obspy.imaging.scripts.scan import main as obspy_scan
from obspy.imaging.scripts.scan import scan, Scanner


class TestScan:
    """
    Test cases for obspy-scan
    """
    root = abspath(join(dirname(__file__), pardir, pardir))
    path = join(root, 'imaging', 'tests', 'images')

    @pytest.fixture(scope='class')
    def all_files(self):
        """return a list of all waveform files in the test suite."""
        sac_files = ['LMOW.BHE.SAC', 'seism.sac', 'dis.G.SCZ.__.BHE_short',
                     'null_terminated.sac', 'test.sac', 'seism-longer.sac',
                     'test.sac.swap', 'seism-shorter.sac', 'testxy.sac']
        gse2_files = ['STA2.testlines', 'STA2.testlines_out', 'acc.gse',
                      'loc_RJOB20050831023349.z',
                      'loc_RJOB20050831023349_first100_dos.z',
                      'loc_RNON20040609200559.z', 'loc_STAU20031119011659.z',
                      'sta2.gse2', 'twiceCHK2.gse2', 'y2000.gse']
        all_files = [join(self.root, 'io', 'sac', 'tests', 'data', i)
                     for i in sac_files]
        all_files.extend([join(self.root, 'io', 'gse2', 'tests', 'data', i)
                          for i in gse2_files])
        return all_files

    def test_scan_main_method(self, all_files, image_path):
        """
        Run obspy-scan on selected tests/data directories
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in all_files:
                shutil.copy(filename, os.curdir)

            obspy_scan([os.curdir] + ['--output', str(image_path)])

    def test_scan_dir_no_permission(self, all_files):
        """
        Run obspy-scan on a directory without read permission.
        Should just skip it and not raise an exception. see #3115
        """
        # Copy files to a temp folder to avoid wildcard scans.
        scanner = Scanner()
        with TemporaryWorkingDirectory():
            no_permission_dir = Path('no_permission_dir')
            no_permission_dir.mkdir()
            # copy one test file in
            shutil.copy(all_files[0], no_permission_dir)
            # take away read permissions
            no_permission_dir.chmod(0o000)
            # still sometimes even on Linux CI runners it seems that stripping
            # read permission does not work and the scanner is able to process
            # it, thus ending up with a higher counter than if properly making
            # the directory not readable
            try:
                os.listdir(no_permission_dir)
            # this is what we want, not being able to read the directory, thus
            # the scanner only processing one item
            except PermissionError:
                pass
            # this is unexpected, we weren't able to strip read permissions,
            # and then in consequence this test is meaningless and we mark it
            # as a skipped test
            else:
                pytest.skip(
                    "unable to remove read permission from a test file for "
                    "testing purposes")
            scanner.parse(str(no_permission_dir))
            # should not have been able to read test file but also not raised
            # an error
            assert scanner.counter == 1
            assert not scanner.data
            # now allow read permission and read again
            no_permission_dir.chmod(0o777)
            scanner.parse(str(no_permission_dir))
            assert scanner.counter == 2
            assert '.LMOW..BHE' in scanner.data
            for child in no_permission_dir.iterdir():
                child.unlink()
            no_permission_dir.rmdir()

    def test_scan_function_and_scanner_class(self, all_files, image_path):
        """
        Test scan function and Scanner class (in one test to keep overhead of
        copying files down)
        """
        scanner = Scanner()
        path1 = image_path.parent / 'scan1.png'
        path2 = image_path.parent / 'scan2.png'
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in all_files:
                shutil.copy(filename, os.curdir)

            scanner.parse(os.curdir)
            scan(paths=os.curdir, plot=path1)

        scanner.plot(path2)

    def test_scan_plot_by_id_with_wildcard(self):
        """
        Test selecting what to plot after scanning with wildcards in selected
        SEED IDs
        """
        files = [
            "BW.UH1._.EHZ.D.2010.147.a.slist.gz",
            "BW.UH1._.EHZ.D.2010.147.b.slist.gz",
            "BW.UH1._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH2._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHE.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHN.D.2010.147.cut.slist.gz",
            "BW.UH3._.SHZ.D.2010.147.cut.slist.gz",
            "BW.UH4._.EHZ.D.2010.147.cut.slist.gz",
            "IUANMO.seed"]

        scanner = Scanner()

        for filename in files:
            scanner.parse(get_example_file(filename))

        expected = [
            ('*.UH[12]*',
             ['BW.UH2..SHZ\n100.0%',
              'BW.UH1..SHZ\n100.0%',
              'BW.UH1..EHZ\n10.7%']),
            ('*Z',
             ['IU.ANMO.00.LHZ\n100.0%',
              'BW.UH4..EHZ\n100.0%',
              'BW.UH3..SHZ\n100.0%',
              'BW.UH2..SHZ\n100.0%',
              'BW.UH1..SHZ\n100.0%',
              'BW.UH1..EHZ\n10.7%'])]

        for seed_id, expected_labels in expected:
            fig, ax = plt.subplots()
            fig = scanner.plot(fig=fig, show=False, seed_ids=[seed_id])
            got = [label.get_text() for label in ax.get_yticklabels()]
            assert got == expected_labels
            plt.close(fig)

    def test_scanner_manually_add_streams(self, all_files, image_path):
        """
        Test Scanner class, manually adding streams of read data files
        """
        scanner = Scanner()
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in all_files:
                shutil.copy(filename, os.curdir)

            for file_ in os.listdir(os.curdir):
                # some files used in the test cases actually can not
                # be read with obspy..
                if file_ in ('STA2.testlines_out', 'STA2.testlines',
                             'seism-shorter.sac', 'seism-longer.sac'):
                    continue
                st = read(file_, headonly=True)
                scanner.add_stream(st)

        scanner.plot(str(image_path))

    def test_scan_save_load_npz(self, all_files, image_path):
        """
        Run obspy-scan on selected tests/data directories, saving/loading
        to/from npz.

        Tests both the command line script and the Scanner class.
        """
        scanner = Scanner()
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in all_files:
                shutil.copy(filename, os.curdir)

            # save via command line
            obspy_scan([os.curdir, '--write', 'scan.npz'])

            # save via Python
            scanner.parse(os.curdir)
            scanner.save_npz('scanner.npz')
            scanner = Scanner()

            path1 = image_path.parent / 'scan_load_npz_1.png'
            path2 = image_path.parent / 'scan_load_npz_2.png'

            # version string of '0.0.0+archive' raises UserWarning - ignore
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore', UserWarning)

                # load via Python
                scanner.load_npz('scanner.npz')
                scanner.plot(path1)

                # load via command line
                obspy_scan(['--load', 'scan.npz', '--output', str(path2)])

    def test_scan_times(self, all_files, image_path):
        """
        Checks for timing related options
        """
        # Copy files to a temp folder to avoid wildcard scans.
        with TemporaryWorkingDirectory():
            for filename in all_files:
                shutil.copy(filename, os.curdir)

            obspy_scan([os.curdir] + ['--output', str(image_path)] +
                       ['--start-time', '2004-01-01'] +
                       ['--end-time', '2004-12-31'] +
                       ['--event-time', '2004-03-14T15:09:26'] +
                       ['--event-time', '2004-02-07T18:28:18'])

    def test_multiple_sampling_rates(self, image_path):
        """
        Check for multiple sampling rates
        """
        lines = [
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:00.000000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R,  50 samples,  50 sps, "
            "2008-01-15T00:00:00.900000, SLIST, INTEGER, Counts",
            "TIMESERIES XX_TEST__BHZ_R, 200 samples, 200 sps, "
            "2008-01-15T00:00:02.000000, SLIST, INTEGER, Counts",
        ]
        files = []
        expected = [
            "XX.TEST..BHZ 2008-01-15T00:00:01.000000Z "
            "2008-01-15T00:00:00.899995Z -0.100",
            "XX.TEST..BHZ 2008-01-15T00:00:01.899999Z "
            "2008-01-15T00:00:02.000000Z 0.100"
        ]
        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2, \
                NamedTemporaryFile() as f3:
            for i, fp in enumerate([f1, f2, f3]):
                fp.write(("%s\n" % lines[i]).encode('ascii',
                                                    'strict'))
                fp.flush()
                fp.seek(0)
                files.append(fp.name)

            # make image comparison instance and set manual rms (see #2089)

            with CatchOutput() as out:
                cmds = ['--output', str(image_path), '--print-gaps']
                obspy_scan(files + cmds)

            # read output and compare with expected
            # only check if datetime objects are close, not exact
            output = out.stdout.splitlines()
            for ex_line, out_line in zip(expected, output):
                ex_split = ex_line.split(' ')
                out_split = out_line.split(' ')
                for ex_str, out_str in zip(ex_split, out_split):
                    try:
                        utc1 = UTCDateTime(ex_str)
                        utc2 = UTCDateTime(out_str)
                    except (ValueError, TypeError):
                        # if str is not a datetime it should be equal
                        assert ex_str == out_str
                    else:
                        # datetimes just need to be close
                        t1, t2 = utc1.timestamp, utc2.timestamp
                        assert abs(t1 - t2) < .001
