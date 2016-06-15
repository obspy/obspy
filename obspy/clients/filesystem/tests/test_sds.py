# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import glob
import imghdr
import inspect
import os
import re
import shutil
import tempfile
import unittest

import numpy as np

from obspy import UTCDateTime, Trace, Stream, read
from obspy.core.util.misc import TemporaryWorkingDirectory, CatchOutput
from obspy.clients.filesystem.sds import SDS_FMTSTR, Client
from obspy.scripts.sds_html_report import main as sds_report


class TemporarySDSDirectory(object):
    """
    Handles creation and deletion of a temporary SDS directory structure.
    To be used with "with" statement.
    """
    sampling_rate = 0.1
    networks = ("AB", "CD")
    stations = ("XYZ", "ZZZ3")
    locations = ("", "00")
    channels = ("HHZ", "HHN", "HHE", "BHZ", "BHN", "BHE")

    def __init__(self, year, doy, time=None):
        """
        Set which day's midnight (00:00 hours) is used as a day break in the
        testing (to split the test data into two files).

        If `time` is specified it overrides `year` and `doy`.
        """
        if time:
            self.time = time
        else:
            self.time = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
        delta = 1.0 / self.sampling_rate

        self.stream = Stream()
        for net in self.networks:
            for sta in self.stations:
                for loc in self.locations:
                    for cha in self.channels:
                        tr = Trace(
                            data=np.arange(100, dtype=np.int32),
                            header=dict(
                                network=net, station=sta, location=loc,
                                channel=cha, sampling_rate=self.sampling_rate,
                                starttime=self.time - 30 * delta))

                        # cut into two seamless traces
                        tr1 = tr.slice(endtime=self.time + 5 * delta)
                        tr2 = tr.slice(starttime=self.time + 6 * delta)
                        self.stream.append(tr1)
                        self.stream.append(tr2)

    def __enter__(self):
        self.old_dir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='obspy-sdstest-')
        for tr_ in self.stream:
            t_ = tr_.stats.starttime
            full_path = SDS_FMTSTR.format(year=t_.year, doy=t_.julday,
                                          sds_type="D", **tr_.stats)
            full_path = os.path.join(self.tempdir, full_path)
            dirname, filename = os.path.split(full_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            tr_.write(full_path, format="MSEED")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        os.chdir(self.old_dir)
        shutil.rmtree(self.tempdir)


class ObsPyTestCase(unittest.TestCase):
    def assert_traces_equal(self, first, second):
        if not isinstance(first, Trace):
            msg = "'first' not a Trace object: " + repr(first)
        if not isinstance(second, Trace):
            msg = "'second' not a Trace object: " + repr(second)
        try:
            np.testing.assert_array_equal(first.data, second.data)
        except Exception as e:
            msg = "Traces' data array not equal:\n" + str(e)
            raise AssertionError(msg)
        try:
            self.assertEqual(first.stats.__dict__, second.stats.__dict__)
        except Exception as e:
            msg = "Traces' Stats not equal:\n" + str(e)
            raise AssertionError(msg)

    def assert_streams_equal(self, first, second):
        if len(first) != len(second):
            msg = "\n".join(("Streams not equal:", str(first), str(second)))
            raise AssertionError(msg)
        msg = "Streams not equal (comparing traces no. {}):\n{}"
        for i, (tr1, tr2) in enumerate(zip(first, second)):
            try:
                self.assert_traces_equal(tr1, tr2)
            except Exception as e:
                msg = msg.format(i, str(e))
                raise AssertionError(msg)

    def assertEqual(self, first, second):  # NoQA
        if isinstance(first, Trace) and isinstance(second, Trace):
            self.assert_traces_equal(first, second)
        if isinstance(first, Stream) and isinstance(second, Stream):
            self.assert_streams_equal(first, second)
        else:
            super(ObsPyTestCase, self).assertEqual(first, second)


class SDSTestCase(ObsPyTestCase):
    """
    Test reading data from SDS file structure.
    """
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read_from_sds(self):
        """
        Test reading data across year and day breaks from SDS directory
        structure. Also tests looking for data on the wrong side of a day
        break (data usually get written for some seconds into the previous or
        next file around midnight).
        """
        # test for one specific SEED ID, without wildcards
        seed_id = "AB.XYZ..HHZ"
        net, sta, loc, cha = seed_id.split(".")
        # use three different day breaks in the testing:
        #  - normal day break during one year
        #     (same directory, separate filenames)
        #  - day break at end of year
        #     (separate directories, separate filenames)
        #      - leap-year
        #      - non-leap-year
        for year, doy in ((2015, 123), (2015, 1), (2012, 1)):
            t = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
            with TemporarySDSDirectory(year=year, doy=doy) as temp_sds:
                # normal test reading across the day break
                client = Client(temp_sds.tempdir)
                st = client.get_waveforms(net, sta, loc, cha, t - 20, t + 20)
                self.assertEqual(len(st), 1)
                self.assertEqual(st[0].stats.starttime, t - 20)
                self.assertEqual(st[0].stats.endtime, t + 20)
                self.assertEqual(len(st[0]), 5)
                # test merge option
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200,
                                          merge=False)
                self.assertEqual(len(st), 2)
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200,
                                          merge=None)
                self.assertEqual(len(st), 2)
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200,
                                          merge=0)
                self.assertEqual(len(st), 1)
                # test reading data from a single day file
                # (data is in the file where it's expected)
                st = client.get_waveforms(net, sta, loc, cha, t - 80, t - 30)
                self.assertEqual(len(st), 1)
                # test reading data from a single day file
                # (data is in the dayfile of the previous day)
                st = client.get_waveforms(net, sta, loc, cha, t + 20, t + 40)
                self.assertEqual(len(st), 1)
                # test that format autodetection with `format=None` works
                client = Client(temp_sds.tempdir, format=None)
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200)
                self.assertEqual(len(st), 1)

    def test_read_from_sds_with_wildcarded_seed_ids(self):
        """
        Test reading data with wildcarded SEED IDs.
        """
        year, doy = 2015, 1
        t = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
        with TemporarySDSDirectory(year=year, doy=doy) as temp_sds:
            # test different wildcard combinations in SEED ID
            client = Client(temp_sds.tempdir)
            for wildcarded_seed_id, num_matching_ids in zip(
                    ("AB.ZZZ3..HH?", "AB.ZZZ3..HH*", "*.*..HHZ",
                     "*.*.*.HHZ", "*.*.*.*"),
                    (3, 3, 4, 8, 48)):
                net, sta, loc, cha = wildcarded_seed_id.split(".")
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200)
                self.assertEqual(len(st), num_matching_ids)
            # test with SDS type wildcards
            for type_wildcard in ("*", "?"):
                net, sta, loc, cha = wildcarded_seed_id.split(".")
                st = client.get_waveforms(net, sta, loc, cha, t - 200, t + 200)
                self.assertEqual(len(st), num_matching_ids)

    def test_sds_report(self):
        """
        Test command line script for generating SDS report html.

        Inherently that script uses many other routines like `_get_filenames`,
        `get_availability_percentage`, `_get_current_endtime`,
        `get_latency`, `has_data` and `get_all_stations`, so these should be
        sufficiently covered as well.
        """
        # generate some dummy SDS with data roughly 2-3 hours behind current
        # time
        t = UTCDateTime() - 2.5 * 3600
        with TemporarySDSDirectory(year=None, doy=None, time=t) as temp_sds, \
                TemporaryWorkingDirectory():
            # create the report
            output_basename = "sds_report"
            argv = [
                "-r={}".format(temp_sds.tempdir),
                "-o={}".format(os.path.join(os.curdir, output_basename)),
                "-l=", "-l=00", "-l=10", "-c=HHZ", "-c=BHZ", "-i=AB.XYZ..BHE",
                "--check-quality-days=1"]
            sds_report(argv)
            # do the testing
            output_basename_abspath = os.path.abspath(
                os.path.join(os.curdir, output_basename))
            file_html = output_basename_abspath + ".html"
            file_txt = output_basename_abspath + ".txt"
            file_png = output_basename_abspath + ".png"
            # check that output files exist
            for file_ in [file_html, file_txt, file_png]:
                self.assertTrue(os.path.isfile(file_))
            # check content of image file (just check it is a png file)
            self.assertEqual(imghdr.what(file_png), "png")
            # check content of stream info / data quality file
            expected_lines = [
                b"AB,XYZ,,BHE,831[0-9].[0-9]*?,0.007292,2",
                b"AB,XYZ,,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"AB,XYZ,00,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"AB,ZZZ3,,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"AB,ZZZ3,00,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"CD,XYZ,,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"CD,XYZ,00,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"CD,ZZZ3,,HHZ,831[0-9].[0-9]*?,0.007292,2",
                b"CD,ZZZ3,00,HHZ,831[0-9].[0-9]*?,0.007292,2"]
            with open(file_txt, "rb") as fh:
                got_lines = fh.readlines()
            for expected_line, got_line in zip(expected_lines, got_lines):
                self.assertIsNotNone(re.match(expected_line, got_line))
            # check content of html report
            with open(file_html, "rb") as fh:
                got_lines = fh.readlines()
            html_regex_file = os.path.join(self.data_dir, "sds_report.regex")
            with open(html_regex_file, "rb") as fh:
                regex_patterns = fh.readlines()
            failed = False  # XXX remove again
            for got, pattern in zip(got_lines, regex_patterns):
                match = re.match(pattern.strip(), got.strip())
                try:
                    self.assertIsNotNone(match)
                except AssertionError:
                    failed = True
                    print(pattern.strip())
                    print(got.strip())
            if failed:
                raise Exception

    def test_get_all_stations_and_nslc(self):
        """
        Test `get_all_stations` and `get_all_nslc` methods
        """
        # generate dummy SDS
        t = UTCDateTime()
        with TemporarySDSDirectory(year=None, doy=None, time=t) as temp_sds:
            client = Client(temp_sds.tempdir)
            expected_netsta = sorted([
                (net, sta)
                for net in temp_sds.networks
                for sta in temp_sds.stations])
            got_netsta = client.get_all_stations()
            self.assertEqual(expected_netsta, got_netsta)
            expected_nslc = sorted([
                (net, sta, loc, cha)
                for net in temp_sds.networks
                for sta in temp_sds.stations
                for loc in temp_sds.locations
                for cha in temp_sds.channels])
            got_nslc = client.get_all_nslc()
            self.assertEqual(expected_nslc, got_nslc)
            got_nslc = client.get_all_nslc(datetime=t)
            self.assertEqual(expected_nslc, got_nslc)
            # other dates that have no data should return empty list
            got_nslc = client.get_all_nslc(datetime=t + 2 * 24 * 3600)
            self.assertEqual([], got_nslc)
            got_nslc = client.get_all_nslc(datetime=t - 2 * 24 * 3600)
            self.assertEqual([], got_nslc)

    def test_add_data_to_archive(self):
        """
        Test `add_data_to_archive` method
        """
        # generate SDS with some gappy data
        t_start = UTCDateTime("2016-180T21")
        t_daybreak1 = UTCDateTime("2016-181")
        t_daybreak2 = UTCDateTime("2016-182")
        tr = Trace(np.arange(1100, dtype=np.int32))
        tr.id = "BW.RMOA..EHZ"
        tr.stats.starttime = t_start
        tr.stats.sampling_rate = 0.01
        st_full = Stream([tr])
        # now cut out some parts and put gappy data in SDS
        st_sds = st_full.copy()
        st_sds.trim(starttime=UTCDateTime("2016-180T22"))
        st_sds.cutout(UTCDateTime("2016-181T10"),
                      UTCDateTime("2016-181T12"))
        st_sds.trim(endtime=UTCDateTime("2016-181T16"))
        st_sds_1 = st_sds.slice(endtime=t_daybreak1-0.1, nearest_sample=False)
        st_sds_2 = st_sds.slice(starttime=t_daybreak1)
        for st in (st_sds_1, st_sds_2):
            for tr in st:
                tr.stats["_format"] = "MSEED"
                tr.stats.pop("processing", None)
        # prepare three streams with new data to add to SDS archive
        st_new_1 = st_full.copy()
        st_new_1.trim(endtime=UTCDateTime("2016-180T23:30")-0.1,
                      nearest_sample=False)
        st_new_1.cutout(UTCDateTime("2016-180T22:20"),
                        UTCDateTime("2016-180T22:50"))
        st_new_2 = st_full.slice(UTCDateTime("2016-181T05"),
                                 UTCDateTime("2016-181T13:30")).copy()
        st_new_2 += st_full.slice(UTCDateTime("2016-181T16"),
                                  UTCDateTime("2016-181T18")).copy()
        st_new_2 += st_full.slice(UTCDateTime("2016-181T20"),
                                  UTCDateTime("2016-181T22")).copy()
        st_new_2 += st_full.slice(
            UTCDateTime("2016-181T23"), t_daybreak2-0.1,
            nearest_sample=False).copy()
        st_new_3 = st_full.slice(t_daybreak2,
                                 UTCDateTime("2016-182T01")).copy()
        # prepare expected streams for the three files
        st_expected_1 = (st_sds_1 + st_new_1).copy()
        st_expected_2 = (st_sds_2 + st_new_2).copy()
        st_expected_3 = st_new_3.copy()
        for st in (st_expected_1, st_expected_2, st_expected_3):
            st.merge(-1)
            for tr in st:
                tr.stats["_format"] = "MSEED"
                tr.stats.pop("processing", None)

        try:
            with TemporaryWorkingDirectory():
                # set up initial SDS structure
                basedir = os.path.abspath(os.path.curdir)
                sds_root = os.path.join(basedir, "SDS")
                filename_1 = os.path.join(sds_root, "2016", "BW", "RMOA",
                                          "EHZ.D", "BW.RMOA..EHZ.D.2016.180")
                filename_2 = os.path.join(sds_root, "2016", "BW", "RMOA",
                                          "EHZ.D", "BW.RMOA..EHZ.D.2016.181")
                filename_3 = os.path.join(sds_root, "2016", "BW", "RMOA",
                                          "EHZ.D", "BW.RMOA..EHZ.D.2016.182")
                os.makedirs(os.path.dirname(filename_1))
                st_sds_1.write(filename_1, format="MSEED")
                st_sds_2.write(filename_2, format="MSEED")
                # setup files with new data
                new_file_1 = os.path.join(basedir, "new_1.slist")
                new_file_2 = os.path.join(basedir, "new_2.mseed")
                st_new_1.write(new_file_1, format="SLIST")
                (st_new_2 + st_new_3).write(new_file_2, format="MSEED")
                # add data and check verbose stdout
                client = Client(sds_root=sds_root)
                with CatchOutput() as out:
                    (new_data_string, changed_files, backupdir,
                     plot_output_file) = client.add_data_to_archive(
                        filenames=glob.glob("new*"))
                self.assertTrue(os.path.exists(plot_output_file))
                for file_ in (filename_1, filename_2, filename_3):
                    self.assertTrue(os.path.exists(file_))
                expected = [
                    "The following files have been appended to:",
                    "\t" + filename_1,
                    "\t" + filename_2,
                    "The following new files have been created:",
                    "\t" + filename_3,
                    "Backups of original files have been stored in: {}".format(
                        backupdir),
                    "Before/after comparison plot saved as: " +
                    plot_output_file,
                    "New data added to archive:",
                    "BW.RMOA..EHZ | 2016-06-28T21:00:00.000000Z - "
                    "2016-06-28T21:58:20.000000Z | 100.0 s, 36 samples",
                    "BW.RMOA..EHZ | 2016-06-29T10:01:40.000000Z - "
                    "2016-06-29T11:58:20.000000Z | 100.0 s, 71 samples",
                    "BW.RMOA..EHZ | 2016-06-29T16:01:40.000000Z - "
                    "2016-06-29T18:00:00.000000Z | 100.0 s, 72 samples",
                    "BW.RMOA..EHZ | 2016-06-29T20:00:00.000000Z - "
                    "2016-06-29T22:00:00.000000Z | 100.0 s, 73 samples",
                    "BW.RMOA..EHZ | 2016-06-29T23:00:00.000000Z - "
                    "2016-06-29T23:58:20.000000Z | 100.0 s, 36 samples",
                    "BW.RMOA..EHZ | 2016-06-30T00:00:00.000000Z - "
                    "2016-06-30T01:00:00.000000Z | 100.0 s, 37 samples",
                    ]
                self.assertEqual(expected, out.stdout.decode().splitlines())
                # check that backup directory holds original data
                for st_orig, doy in zip((st_sds_1, st_sds_2), ("180", "181")):
                    st_backup = read(os.path.join(
                        backupdir, "2016", "BW", "RMOA", "EHZ.D",
                        "BW.RMOA..EHZ.D.2016.{}".format(doy)), format="MSEED")
                    for tr in st_backup:
                        tr.stats.pop("mseed", None)
                    self.assertEqual(st_orig, st_backup)
                # now check the contents of SDS after adding the data
                st_got_1 = read(filename_1, format="MSEED")
                st_got_2 = read(filename_2, format="MSEED")
                st_got_3 = read(filename_3, format="MSEED")
                for st in (st_got_1, st_got_2, st_got_3):
                    for tr in st:
                        tr.stats.pop("mseed")
                    st.merge(-1)
                for expected, got in zip(
                        (st_expected_1, st_expected_2, st_expected_3),
                        (st_got_1, st_got_2, st_got_3)):
                    self.assertEqual(expected, got)
                # finally, add some data that ends up in a file whose directory
                # does not exist yet, first of an existing SEED ID (before and
                # after any other existing data) and then with a completely new
                # SEED ID
                tr = Trace(np.ones(10))
                got_stdout = []
                expected_trace_infos = []
                comparison_plot_filename = os.path.join(
                    tempfile.gettempdir(), "obspy-sds-testplot.png")
                for id in ("BW.RMOA..EHZ", "XX.XYZ.99.KLM"):
                    for t in ("1980-003", "2100-345"):
                        tr.id = id
                        tr.stats.starttime = t
                        expected_trace_infos.append(str(tr).splitlines()[-1])
                        filename = os.path.join(basedir, "other.slist")
                        tr.write(filename, format="SLIST")
                        with CatchOutput() as out:
                            _, _, _, plot_output_file_2 = \
                                client.add_data_to_archive(
                                    filenames=(filename,),
                                    plot=comparison_plot_filename)
                        got_stdout += out.stdout.decode().splitlines()
                self.assertEqual(comparison_plot_filename, plot_output_file_2)
                self.assertTrue(os.path.exists(comparison_plot_filename))
                expected_paths = (
                    ("1980", "BW", "RMOA", "EHZ.D", "BW.RMOA..EHZ.D.1980.003"),
                    ("2100", "BW", "RMOA", "EHZ.D", "BW.RMOA..EHZ.D.2100.345"),
                    ("1980", "XX", "XYZ", "KLM.D", "XX.XYZ.99.KLM.D.1980.003"),
                    ("2100", "XX", "XYZ", "KLM.D", "XX.XYZ.99.KLM.D.2100.345"))
                expected_files = [os.path.join(sds_root, *expected_path)
                                  for expected_path in expected_paths]
                expected_stdout = [
                    "The following new files have been created:",
                    "\t" + expected_files[0],
                    "Before/after comparison plot saved as: " +
                    comparison_plot_filename,
                    "New data added to archive:",
                    expected_trace_infos[0],
                    "The following new files have been created:",
                    "\t" + expected_files[1],
                    "Before/after comparison plot saved as: " +
                    comparison_plot_filename,
                    "New data added to archive:",
                    expected_trace_infos[1],
                    "The following new files have been created:",
                    "\t" + expected_files[2],
                    "Before/after comparison plot saved as: " +
                    comparison_plot_filename,
                    "New data added to archive:",
                    expected_trace_infos[2],
                    "The following new files have been created:",
                    "\t" + expected_files[3],
                    "Before/after comparison plot saved as: " +
                    comparison_plot_filename,
                    "New data added to archive:",
                    expected_trace_infos[3],
                    ]
                self.assertEqual(expected_stdout, got_stdout)
                for expected_file in expected_files:
                    self.assertTrue(os.path.isfile(expected_file))
                self.assertTrue(os.path.isdir(backupdir))
        finally:
            try:
                os.remove(plot_output_file)
            except:
                pass
            try:
                os.remove(comparison_plot_filename)
            except:
                pass
            try:
                shutil.rmtree(backupdir)
            except:
                pass


def suite():
    return unittest.makeSuite(SDSTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
