# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest
from obspy.core.compatibility import mock

import obspy
from obspy import UTCDateTime
from obspy.clients.filesystem.tsindex import Client

from collections import namedtuple
import os


class TSIndexTestCase(unittest.TestCase):

    def test_get_waveforms(self):
        package_dir = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(package_dir, 'data/tsindex_data/')
        db_path = os.path.join(filepath, 'timeseries.sqlite')

        # part of one file
        client = Client(db_path,
                        datapath_replace=(None,
                                          filepath))
        returned_stream = client.get_waveforms(
                                  "IU", "ANMO", "10", "BHZ",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        expected_stream = \
            obspy.read(filepath + 'IU.ANMO.10.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # wildcard request spanning mutliple files
        returned_stream = client.get_waveforms(
                                  "*", "*", "00,10", "BHZ",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream1 = \
            obspy.read(filepath + 'IU.ANMO.10.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream2 = \
            obspy.read(filepath + 'IU.COLA.10.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream3 = \
            obspy.read(filepath + 'CU.TGUH.00.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream = \
            expected_stream1 + expected_stream2 + expected_stream3
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # no data
        returned_stream = client.get_waveforms(
                                  "XX", "XXX", "XX", "XXX",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        self.assertListEqual(returned_stream.traces, [])

    def test_get_waveforms_bulk(self):
        package_dir = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(package_dir, 'data/tsindex_data/')
        db_path = os.path.join(filepath, 'timeseries.sqlite')

        # part of one file
        client = Client(db_path,
                        datapath_replace=(None,
                                          filepath))
        bulk_request = [
                        ("IU", "ANMO", "10", "BHZ",
                         UTCDateTime(2018, 1, 1, 0, 0, 0),
                         UTCDateTime(2018, 1, 1, 0, 0, 5)),
                        ("CU", "TGUH", "00", "BHZ",
                         UTCDateTime(2018, 1, 1, 0, 0, 1),
                         UTCDateTime(2018, 1, 1, 0, 0, 7)),
                        ]
        returned_stream = client.get_waveforms_bulk(bulk_request)
        expected_stream1 = \
            obspy.read(filepath + 'IU.ANMO.10.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        expected_stream2 = \
            obspy.read(filepath + 'CU.TGUH.00.BHZ.2018.001_first_minute.mseed',
                       starttime=UTCDateTime(2018, 1, 1, 0, 0, 1),
                       endtime=UTCDateTime(2018, 1, 1, 0, 0, 7))
        expected_stream = expected_stream1 + expected_stream2
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # no data
        returned_stream = client.get_waveforms(
                                  "XX", "XXX", "XX", "XXX",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        self.assertListEqual(returned_stream.traces, [])

    def test_get_nslc(self):
        client = Client("")

        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest'])
        mocked_summary_rows = [NamedRow("AK", "ANM", "", "VM2",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM3",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM4",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM5",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415")]
        client.get_summary_rows = mock.MagicMock(
                                            return_value=mocked_summary_rows)

        expected_avail_extents = [("AK", "ANM", "", "VM2"),
                                  ("AK", "ANM", "", "VM3"),
                                  ("AK", "ANM", "", "VM4"),
                                  ("AK", "ANM", "", "VM5"),
                                  ("N4", "H43A", "", "VM2"),
                                  ("N4", "H43A", "", "VM3")]

        self.assertEqual(client.get_nslc("AK,N4",
                                         "ANM,H43A",
                                         "",
                                         "VM2,VM3,VM4,VM5",
                                         "2018-08-10T21:09:39.000000",
                                         "2018-08-10T22:09:28.890415"),
                         expected_avail_extents)

    def test_get_availability_extent(self):
        client = Client("")

        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest'])
        mocked_summary_rows = [NamedRow("AK", "ANM", "", "VM2",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM3",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM4",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM5",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415")]
        client.get_summary_rows = mock.MagicMock(
                                            return_value=mocked_summary_rows)

        expected_avail_extents = [("AK", "ANM", "", "VM2",
                                   UTCDateTime("2018-08-10T21:52:50.000000"),
                                   UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM3",
                                   UTCDateTime("2018-08-10T21:52:50.000000"),
                                   UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM4",
                                   UTCDateTime("2018-08-10T21:52:50.000000"),
                                   UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("AK", "ANM", "", "VM5",
                                   UTCDateTime("2018-08-10T21:52:50.000000"),
                                   UTCDateTime("2018-08-10T22:12:39.999991")),
                                  ("N4", "H43A", "", "VM2",
                                   UTCDateTime("2018-08-10T21:09:39.000000"),
                                   UTCDateTime("2018-08-10T22:09:28.890415")),
                                  ("N4", "H43A", "", "VM3",
                                   UTCDateTime("2018-08-10T21:09:39.000000"),
                                   UTCDateTime("2018-08-10T22:09:28.890415"))]

        self.assertEqual(client.get_availability_extent(
                                                "AK,N4",
                                                "ANM,H43A", "",
                                                "VM2,VM3,VM4,VM5",
                                                "2018-08-10T21:09:39.000000",
                                                "2018-08-10T22:09:28.890415"),
                         expected_avail_extents)

    def test__are_timespans_adjacent(self):
        client = Client("")
        sample_rate = 40
        # sample_period = 1/40 = 0.025 sec = 25000 ms
        # and tolerance = 0.5 so an adjacent sample is +/-0.0125 sec = 12500 ms

        # 1 sample period later
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 50000),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # 1ms after nearest tolerance boundary (next sample - 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37501),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # exactly on nearest tolerance boundary (next sample - 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37500),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # 1ms before nearest tolerance boundary (next sample - 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 37499),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 75000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # 1ms after farthest tolerance boundary (next sample + 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62501),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # on farthest tolerance boundary (next sample + 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62500),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertFalse(client._are_timespans_adjacent(ts1, ts2, sample_rate))

        # 1ms before farthest tolerance boundary (next sample + 12500ms)
        ts1 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 0),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 25000))
        ts2 = \
            client._create_timespan(UTCDateTime(2018, 8, 10, 22, 0, 0, 62499),
                                    UTCDateTime(2018, 8, 10, 22, 0, 0, 100000))
        self.assertTrue(client._are_timespans_adjacent(ts1, ts2, sample_rate))

    def test_get_availability(self):
        client = Client("")

        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'samplerate', 'starttime', 'endtime',
                               'timespans'])
        # Each row contains one gap. The first rows latest time is adjacent
        # to the second rows earliest time. The third row overlaps with the
        # second row.
        mocked_tsindex_rows = \
            [
               # 2018-08-10T22:00:54 to 2018-08-10T22:15:53
               # 2018-08-10T22:05:54 to 2018-08-10T22:20:53 (MERGE)
               NamedRow(network=u'AK', station=u'BAGL',
                        location=u'', channel=u'LCC',
                        starttime=u'2018-08-10T22:00:54.000000',
                        endtime=u'2018-08-10T22:20:53.000000',
                        samplerate=1.0,
                        timespans=u'[1533938454.000000:1533939353.000000],'
                                  u'[1533938754.000000:1533939653.000000]'),
               # 2018-08-10T22:20:53.999000 to 2018-08-12T22:20:53 (JOIN)
               # 2018-08-12T23:20:53 to 2018-09-01T23:20:53
               NamedRow(network=u'AK', station=u'BAGL',
                        location=u'', channel=u'LCC',
                        starttime=u'2018-08-10T22:20:53.999000',
                        endtime=u'2018-09-01T23:20:53.000000',
                        samplerate=1.0,
                        timespans=u'[1533939653.999000:1534112453.000000],'
                                  u'[1534116053.000000:1535844053.000000]'),
               # 2018-08-27T00:00:00 to 2018-09-11T00:00:00 (MERGE)
               NamedRow(network=u'AK', station=u'BAGL',
                        location=u'', channel=u'LCC',
                        starttime=u'2018-08-27T00:00:00.000000',
                        endtime=u'2018-09-11T00:00:00.000000',
                        samplerate=1.0,
                        timespans=u'[1535328000.0:1536624000.0]')
            ]
        client.get_tsindex_rows = mock.MagicMock(
                                            return_value=mocked_tsindex_rows)

        expected_unmerged_avail = [("AK", "BAGL", "", "LCC",
                                    UTCDateTime(2018, 8, 10, 22, 0, 54),
                                    UTCDateTime(2018, 8, 10, 22, 15, 53)),
                                   ("AK", "BAGL", "", "LCC",
                                    UTCDateTime(2018, 8, 10, 22, 5, 54),
                                    UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                   ("AK", "BAGL", "", "LCC",
                                    UTCDateTime(2018, 8, 12, 23, 20, 53),
                                    UTCDateTime(2018, 9, 1, 23, 20, 53)),
                                   ("AK", "BAGL", "", "LCC",
                                    UTCDateTime(2018, 8, 27, 0, 0),
                                    UTCDateTime(2018, 9, 11, 0, 0, 0))]

        self.assertEqual(client.get_availability(
                                "AK",
                                "BAGL", "",
                                "LCC",
                                UTCDateTime(2018, 8, 10, 22, 0, 54),
                                UTCDateTime(2018, 8, 10, 22, 9, 28, 890415)),
                         expected_unmerged_avail)

        self.assertEqual(client.get_availability(
                                "AK",
                                "BAGL",
                                "--",
                                "LCC",
                                UTCDateTime(2018, 8, 10, 22, 0, 54),
                                UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                                merge_overlap=False),
                         expected_unmerged_avail)

        expected_merged_avail = [("AK", "BAGL", "", "LCC",
                                  UTCDateTime(2018, 8, 10, 22, 0, 54),
                                  UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                 ("AK", "BAGL", "", "LCC",
                                  UTCDateTime(2018, 8, 12, 23, 20, 53),
                                  UTCDateTime(2018, 9, 11, 0, 0, 0))]

        self.assertEqual(client.get_availability(
                                "AK",
                                "BAGL",
                                "--",
                                "LCC",
                                UTCDateTime(2018, 8, 10, 22, 0, 54),
                                UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                                merge_overlap=True),
                         expected_merged_avail)

    def test_get_availability_percentage(self):
        client = Client("")

        mock_availability_output = [("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 8, 10, 22, 0, 54),
                                     UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                    ("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 8, 12, 23, 20, 53),
                                     UTCDateTime(2018, 9, 11, 0, 0, 0))]
        client.get_availability = \
            mock.MagicMock(return_value=mock_availability_output)

        avail_percentage = client.get_availability_percentage(
                                        "AK",
                                        "BAGL",
                                        "--",
                                        "LCC",
                                        UTCDateTime(2018, 8, 10, 22, 0, 54),
                                        UTCDateTime(2018, 9, 11, 0, 0, 0))
        expected_avail_percentage = (0.998659490472, 1)
        self.assertAlmostEqual(avail_percentage[0],
                               expected_avail_percentage[0])
        self.assertEqual(avail_percentage[1],
                         expected_avail_percentage[1])
        self.assertIsInstance(avail_percentage, tuple)

        mock_availability_output = [("AF", "SOE", "", "BHE",
                                     UTCDateTime(2018, 1, 1),
                                     UTCDateTime(2018, 1, 2)),
                                    ("AF", "SOE", "", "BHE",
                                     UTCDateTime(2018, 1, 3),
                                     UTCDateTime(2018, 1, 4)),
                                    ("AF", "SOE", "", "BHE",
                                     UTCDateTime(2018, 1, 5),
                                     UTCDateTime(2018, 1, 6))]

        client.get_availability = \
            mock.MagicMock(return_value=mock_availability_output)

        avail_percentage = client.get_availability_percentage(
                                                    "AK",
                                                    "BAGL",
                                                    "--",
                                                    "LCC",
                                                    UTCDateTime(2018, 1, 1),
                                                    UTCDateTime(2018, 1, 6))
        expected_avail_percentage = (0.6, 2)
        self.assertAlmostEqual(avail_percentage[0],
                               expected_avail_percentage[0])
        self.assertEqual(avail_percentage[1],
                         expected_avail_percentage[1])
        self.assertIsInstance(avail_percentage, tuple)

        # Test for over extending time span
        avail_percentage = client.get_availability_percentage(
                                                    "AK",
                                                    "BAGL",
                                                    "--",
                                                    "LCC",
                                                    UTCDateTime(2017, 12, 31),
                                                    UTCDateTime(2018, 1, 7))
        expected_avail_percentage = (0.4285714, 4)
        self.assertAlmostEqual(avail_percentage[0],
                               expected_avail_percentage[0])
        self.assertEqual(avail_percentage[1],
                         expected_avail_percentage[1])
        self.assertIsInstance(avail_percentage, tuple)
