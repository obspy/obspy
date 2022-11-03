# -*- coding: utf-8 -*-
import inspect
import os
import re
import shutil
import tempfile
import unittest

import numpy as np

from obspy import UTCDateTime, Trace, Stream
from obspy.core.util.misc import TemporaryWorkingDirectory
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


class SDSTestCase(unittest.TestCase):
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
            # look for png static header
            with open(file_png, 'rb') as fh:
                assert fh.read(8) == b'\x89PNG\r\n\x1a\n'
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

    def test_get_waveforms_bulk(self):
        """
        Test get_waveforms_bulk method.
        """
        year = 2015
        doy = 247
        t = UTCDateTime("%d-%03dT00:00:00" % (year, doy))
        with TemporarySDSDirectory(year=year, doy=doy) as temp_sds:
            chunks = [
                ["AB", "XYZ", "", "HHZ", t, t + 20],
                ["AB", "XYZ", "", "HHN", t + 20, t + 40],
                ["AB", "XYZ", "", "HHE", t + 40, t + 60],
                ["CD", "ZZZ3", "00", "BHZ", t + 60, t + 80],
                ["CD", "ZZZ3", "00", "BHN", t + 80, t + 100],
                ["CD", "ZZZ3", "00", "BHE", t + 120, t + 140]
            ]
            client = Client(temp_sds.tempdir)
            st = client.get_waveforms_bulk(chunks)
            for _i in range(6):
                if _i <= 2:
                    self.assertEqual(st[_i].stats.network, "AB")
                    self.assertEqual(st[_i].stats.station, "XYZ")
                    self.assertEqual(st[_i].stats.location, "")
                elif _i >= 2:
                    self.assertEqual(st[_i].stats.network, "CD")
                    self.assertEqual(st[_i].stats.station, "ZZZ3")
                    self.assertEqual(st[_i].stats.location, "00")
            self.assertEqual(st[0].stats.channel, "HHZ")
            self.assertEqual(st[1].stats.channel, "HHN")
            self.assertEqual(st[2].stats.channel, "HHE")
            self.assertEqual(st[3].stats.channel, "BHZ")
            self.assertEqual(st[4].stats.channel, "BHN")
            self.assertEqual(st[5].stats.channel, "BHE")

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
