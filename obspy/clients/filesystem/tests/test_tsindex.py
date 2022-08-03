# -*- coding: utf-8 -*-
from collections import namedtuple

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import os
import re
import requests
import tempfile
import unittest
import uuid
from unittest import mock, TestCase

from obspy.clients.filesystem.tsindex import Client, Indexer, \
    TSIndexDatabaseHandler
from obspy import read
from obspy import UTCDateTime


def get_test_data_filepath():
    package_dir = os.path.abspath(os.path.dirname(__file__))
    # TODO It perhaps shouldn't be the case, but the string here has to end
    # in a forward slash. We should fix this in the future.
    filepath = os.path.join(package_dir, 'data', 'tsindex_data/')
    return filepath


def get_test_client():
    filepath = get_test_data_filepath()
    db_path = os.path.join(filepath, 'timeseries.sqlite')

    client = Client(db_path,
                    datapath_replace=("^",
                                      filepath))
    return client


class ClientTestCase(TestCase):

    def test_bad_sqlitdb_filepath(self):
        """
        Checks that an error is raised when an invalid path is provided to
        a SQLite database
        """
        self.assertRaisesRegex(OSError,
                               "^Database path.*does not exist.$",
                               Client,
                               "/some/bad/path/timeseries.sqlite")

    def test_get_waveforms(self):
        filepath = get_test_data_filepath()
        client = get_test_client()

        expected_stream = \
            read(
                filepath + os.path.join(
                    'IU', '2018', '001',
                    'IU.ANMO.10.BHZ.2018.001_first_minute.mseed'),
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        returned_stream = client.get_waveforms(
            "IU", "ANMO", "10", "BHZ",
            starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
            endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        returned_stream.sort()
        expected_stream.sort()

        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # wildcard request spanning multiple files
        expected_stream1 = \
            read(
                filepath + os.path.join(
                    'CU', '2018', '001',
                    'CU.TGUH.00.BHZ.2018.001_first_minute.mseed'),
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream2 = \
            read(
                filepath + os.path.join(
                    'IU', '2018', '001',
                    'IU.ANMO.10.BHZ.2018.001_first_minute.mseed'),
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream3 = \
            read(
                filepath + os.path.join(
                    'IU', '2018', '001',
                    'IU.COLA.10.BHZ.2018.001_first_minute.mseed'),
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        expected_stream = \
            expected_stream1 + expected_stream2 + expected_stream3
        returned_stream = client.get_waveforms(
                                  "*", "*", "10,00", "BHZ",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        returned_stream.sort()
        expected_stream.sort()

        self.assertEqual(len(returned_stream), len(expected_stream))
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # request resulting in no data
        returned_stream = client.get_waveforms(
                                  "XX", "XXX", "XX", "XXX",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        self.assertListEqual(returned_stream.traces, [])

    def test_get_waveforms_bulk(self):
        filepath = get_test_data_filepath()
        client = get_test_client()

        expected_stream1 = \
            read(
                filepath +
                'CU/2018/001/CU.TGUH.00.BHZ.2018.001_first_minute.mseed',
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 1),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 7))
        expected_stream2 = \
            read(
                filepath +
                'IU/2018/001/IU.ANMO.10.BHZ.2018.001_first_minute.mseed',
                starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                endtime=UTCDateTime(2018, 1, 1, 0, 0, 5))
        expected_stream = expected_stream1 + expected_stream2
        expected_stream.sort()

        # assert equal
        bulk_request = [
                        ("CU", "TGUH", "00", "BHZ",
                         UTCDateTime(2018, 1, 1, 0, 0, 1),
                         UTCDateTime(2018, 1, 1, 0, 0, 7)),
                        ("IU", "ANMO", "10", "BHZ",
                         UTCDateTime(2018, 1, 1, 0, 0, 0),
                         UTCDateTime(2018, 1, 1, 0, 0, 5)),
                        ]
        returned_stream = client.get_waveforms_bulk(bulk_request)
        returned_stream.sort()

        self.assertEqual(len(returned_stream), len(expected_stream))
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertListEqual(list(t1.data), list(t2.data))

        # assert not equal with a non-equivalent request
        bulk_request = [
                        ("IU", "ANMO", "10", "BHZ",
                         UTCDateTime(2018, 1, 1, 0, 0, 0),
                         UTCDateTime(2018, 1, 1, 0, 0, 5)),
                        ]
        returned_stream = client.get_waveforms_bulk(bulk_request)
        returned_stream.sort()
        self.assertNotEqual(len(returned_stream), len(expected_stream))
        for t1, t2 in zip(returned_stream, expected_stream):
            self.assertRaises(AssertionError,
                              self.assertListEqual,
                              list(t1.data), list(t2.data))

        # request resulting in no data
        returned_stream = client.get_waveforms(
                                  "XX", "XXX", "XX", "XXX",
                                  starttime=UTCDateTime(2018, 1, 1, 0, 0, 0),
                                  endtime=UTCDateTime(2018, 1, 1, 0, 0, 3, 1))
        self.assertListEqual(returned_stream.traces, [])

    def test_get_nslc(self):
        client = get_test_client()
        # test using actual sqlite3 test database
        expected_nslc = [(u'CU', u'TGUH', u'00', u'BHZ')]

        actual_nslc = client.get_nslc("I*,CU",
                                      "ANMO,COL?,T*",
                                      "00,10",
                                      "BHZ",
                                      "2018-01-01T00:00:00.000000",
                                      "2018-01-01T00:00:00.019499")
        self.assertListEqual(actual_nslc, expected_nslc)
        actual_nslc = client.get_nslc("CU",
                                      "ANMO,COL?,T*")
        self.assertListEqual(actual_nslc, expected_nslc)

        # test using mocked client._get_summary_rows method for more diversity
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
                               NamedRow("XX", "ANM", "", "VM5",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415")]
        client._get_summary_rows = \
            mock.MagicMock(return_value=mocked_summary_rows)

        expected_nslc = [("AK", "ANM", "", "VM2"),
                         ("AK", "ANM", "", "VM3"),
                         ("AK", "ANM", "", "VM4"),
                         ("N4", "H43A", "", "VM2"),
                         ("N4", "H43A", "", "VM3"),
                         ("XX", "ANM", "", "VM5")]

        self.assertEqual(client.get_nslc("AK,N4,XX",
                                         "ANM,H43A",
                                         "",
                                         "VM2,VM3,VM4,VM5",
                                         "2018-08-10T21:09:39.000000",
                                         "2018-08-10T22:09:28.890415"),
                         expected_nslc)

    def test_get_availability_extent(self):
        client = get_test_client()
        # test using actual sqlite test database
        expected_nslc = [(u'IU', u'ANMO', u'10', u'BHZ',
                          UTCDateTime(2018, 1, 1, 0, 0, 0, 19500),
                          UTCDateTime(2018, 1, 1, 0, 0, 59, 994536)),
                         (u'IU', u'COLA', u'10', u'BHZ',
                          UTCDateTime(2018, 1, 1, 0, 0, 0, 19500),
                          UTCDateTime(2018, 1, 1, 0, 0, 59, 994538))]

        actual_avail_extents = client.get_availability_extent(
                                                "I*",
                                                "ANMO,COL?,T*",
                                                "00,10",
                                                "BHZ",
                                                "2018-01-01T00:00:00.000000",
                                                "2018-12-31T00:00:00.000000")
        self.assertListEqual(actual_avail_extents, expected_nslc)

        actual_avail_extents = client.get_availability_extent("I*")
        self.assertListEqual(actual_avail_extents, expected_nslc)

        # test using mocked client._get_summary_rows method for more diversity
        NamedRow = namedtuple('NamedRow',
                              ['network', 'station', 'location', 'channel',
                               'earliest', 'latest'])
        mocked_summary_rows = [NamedRow("AK", "ANM", "", "VM2",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM3",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991"),
                               NamedRow("AK", "ANM", "", "VM5",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:15:39.999991"),
                               NamedRow("N4", "H43A", "", "VM2",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415"),
                               NamedRow("N4", "H43A", "", "VM3",
                                        "2018-08-10T21:09:39.000000",
                                        "2018-08-10T22:09:28.890415"),
                               NamedRow("XX", "ANM", "", "VM4",
                                        "2018-08-10T21:52:50.000000",
                                        "2018-08-10T22:12:39.999991")]
        client._get_summary_rows = \
            mock.MagicMock(return_value=mocked_summary_rows)

        expected_avail_extents = \
            [("AK", "ANM", "", "VM2",
              UTCDateTime("2018-08-10T21:52:50.000000"),
              UTCDateTime("2018-08-10T22:12:39.999991")),
             ("AK", "ANM", "", "VM3",
              UTCDateTime("2018-08-10T21:52:50.000000"),
              UTCDateTime("2018-08-10T22:12:39.999991")),
             ("AK", "ANM", "", "VM5",
              UTCDateTime("2018-08-10T21:52:50.000000"),
              UTCDateTime("2018-08-10T22:15:39.999991")),
             ("N4", "H43A", "", "VM2",
              UTCDateTime("2018-08-10T21:09:39.000000"),
              UTCDateTime("2018-08-10T22:09:28.890415")),
             ("N4", "H43A", "", "VM3",
              UTCDateTime("2018-08-10T21:09:39.000000"),
              UTCDateTime("2018-08-10T22:09:28.890415")),
             ("XX", "ANM", "", "VM4",
              UTCDateTime("2018-08-10T21:52:50.000000"),
              UTCDateTime("2018-08-10T22:12:39.999991"))]

        self.assertListEqual(client.get_availability_extent(
                                            "AK,N4",
                                            "ANM,H43A", "",
                                            "VM2,VM3,VM4,VM5",
                                            "2018-08-10T21:09:39.000000",
                                            "2018-08-10T22:09:28.890415"),
                             expected_avail_extents)

    def test__are_timespans_adjacent(self):
        client = get_test_client()

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
        client = get_test_client()
        # test using actual sqlite test database
        expected_avail = [(u'IU', u'ANMO', u'10', u'BHZ',
                           UTCDateTime(2018, 1, 1, 0, 0, 0, 19500),
                           UTCDateTime(2018, 1, 1, 0, 0, 59, 994536)),
                          (u'IU', u'COLA', u'10', u'BHZ',
                           UTCDateTime(2018, 1, 1, 0, 0, 0, 19500),
                           UTCDateTime(2018, 1, 1, 0, 0, 59, 994538))]

        actual_avail = client.get_availability(
                                                "IU",
                                                "ANMO,COLA",
                                                "10",
                                                "BHZ",
                                                "2018-01-01",
                                                "2018-12-31")
        self.assertListEqual(actual_avail, expected_avail)

        actual_avail = client.get_availability("IU",
                                               "ANMO,COLA")
        self.assertListEqual(actual_avail, expected_avail)

        # test using mocked client._get_summary_rows method for more diversity
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
               NamedRow(
                   network=u'AK', station=u'BAGL',
                   location=u'', channel=u'LCC',
                   starttime=u'2018-08-10T22:00:54.000000',
                   endtime=u'2018-08-10T22:20:53.000000',
                   samplerate=1.0,
                   timespans=u'[1533938454.000000:1533939353.000000],'
                             u'[1533938754.000000:1533939653.000000]'),
               # 2018-08-10T22:20:53.999000 to 2018-08-12T22:20:53 (JOIN)
               # 2018-08-12T23:20:53 to 2018-09-01T23:20:53
               NamedRow(
                   network=u'AK', station=u'BAGL',
                   location=u'', channel=u'LCC',
                   starttime=u'2018-08-10T22:20:53.999000',
                   endtime=u'2018-09-01T23:20:53.000000',
                   samplerate=1.0,
                   timespans=u'[1533939653.999000:1534112453.000000],'
                             u'[1534116053.000000:1535844053.000000]'),
               # (MERGE IF INCL SAMPLE RATE IS TRUE)
               # 2018-08-27T00:00:00 to 2018-09-11T00:00:00
               NamedRow(network=u'AK', station=u'BAGL',
                        location=u'', channel=u'LCC',
                        starttime=u'2018-08-27T00:00:00.000000',
                        endtime=u'2018-09-11T00:00:00.000000',
                        samplerate=10.0,
                        timespans=u'[1535328000.0:1536624000.0]')
            ]
        client._get_tsindex_rows = \
            mock.MagicMock(return_value=mocked_tsindex_rows)

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

        # test default options
        self.assertListEqual(client.get_availability(
                            "AK",
                            "BAGL",
                            "",
                            "LCC",
                            UTCDateTime(2018, 8, 10, 22, 0, 54),
                            UTCDateTime(2018, 8, 10, 22, 9, 28, 890415)),
                         expected_unmerged_avail)

        # test merge overlap false
        self.assertListEqual(client.get_availability(
                            "AK",
                            "BAGL",
                            "--",
                            "LCC",
                            UTCDateTime(2018, 8, 10, 22, 0, 54),
                            UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                            merge_overlap=False),
                         expected_unmerged_avail)

        # test merge overlap true
        expected_merged_avail = [("AK", "BAGL", "", "LCC",
                                  UTCDateTime(2018, 8, 10, 22, 0, 54),
                                  UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                 ("AK", "BAGL", "", "LCC",
                                  UTCDateTime(2018, 8, 12, 23, 20, 53),
                                  UTCDateTime(2018, 9, 11, 0, 0, 0))]

        self.assertListEqual(
                client.get_availability(
                    "AK",
                    "BAGL",
                    "--",
                    "LCC",
                    UTCDateTime(2018, 8, 10, 22, 0, 54),
                    UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                    merge_overlap=True),
                expected_merged_avail)

        # test include_sample_rate true
        expected_incl_sr_avail = \
            [("AK", "BAGL", "", "LCC",
              UTCDateTime(2018, 8, 10, 22, 0, 54),
              UTCDateTime(2018, 8, 12, 22, 20, 53), 1.0),
             ("AK", "BAGL", "", "LCC",
              UTCDateTime(2018, 8, 12, 23, 20, 53),
              UTCDateTime(2018, 9, 1, 23, 20, 53), 1.0),
             ("AK", "BAGL", "", "LCC",
              UTCDateTime(2018, 8, 27, 0, 0),
              UTCDateTime(2018, 9, 11, 0, 0, 0), 10.0)]

        self.assertListEqual(
                client.get_availability(
                    "AK",
                    "BAGL",
                    "--",
                    "LCC",
                    UTCDateTime(2018, 8, 10, 22, 0, 54),
                    UTCDateTime(2018, 8, 10, 22, 9, 28, 890415),
                    merge_overlap=True,
                    include_sample_rate=True),
                expected_incl_sr_avail)

    def test_get_availability_percentage(self):
        client = get_test_client()
        mock_availability_output = [("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 8, 10, 22, 0, 54),
                                     UTCDateTime(2018, 8, 12, 22, 20, 53)),
                                    ("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 8, 12, 23, 20, 53),
                                     UTCDateTime(2018, 9, 11, 0, 0, 0))]
        client.get_availability = mock.MagicMock(
                                        return_value=mock_availability_output)

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

        mock_availability_output = [("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 1, 1),
                                     UTCDateTime(2018, 1, 2)),
                                    ("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 1, 3),
                                     UTCDateTime(2018, 1, 4)),
                                    ("AK", "BAGL", "", "LCC",
                                     UTCDateTime(2018, 1, 5),
                                     UTCDateTime(2018, 1, 6))]
        client.get_availability = mock.MagicMock(
                                        return_value=mock_availability_output)

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

    def test_has_data(self):
        client = get_test_client()
        self.assertTrue(client.has_data())
        self.assertTrue(client.has_data(starttime=UTCDateTime(2017, 12, 31),
                                        endtime=UTCDateTime(2018, 1, 7)))
        self.assertFalse(client.has_data(starttime=UTCDateTime(1970, 12, 31),
                                         endtime=UTCDateTime(2013, 1, 7)))


def purge(dir, pattern):
    """
    Delete a file matching a pattern from the OS.
    """
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


class IndexerTestCase(TestCase):

    def test_bad_rootpath(self):
        """
        Checks that an error is raised when an invalid path is provided to
        a data
        """
        filepath = get_test_data_filepath()
        database = os.path.join(filepath,
                                'timeseries.sqlite')

        # test that a bad leap second file path raises an error
        self.assertRaisesRegex(OSError,
                               "^Root path.*does not exists.$",
                               Indexer,
                               "/some/bad/path",
                               database=database,
                               filename_pattern="*.mseed",
                               parallel=2)

    def test_bad_sqlitdb_filepath(self):
        """
        Checks that an error is raised when an invalid path is provided to
        a SQLite database
        """
        filepath = get_test_data_filepath()
        self.assertRaisesRegex(OSError,
                               "^Database path.*does not exist.$",
                               Indexer,
                               filepath,
                               database='/some/bad/path/',
                               filename_pattern="*.mseed")

    def test_bad_database(self):
        """
        Checks that an error is raised when an invalid database is provided.
        The database must be a TSIndexDatabaseHandler or a str, otherwise
        a ValueError is raised.
        """
        filepath = get_test_data_filepath()
        self.assertRaisesRegex(ValueError,
                               "^Database must be a string or "
                               "TSIndexDatabaseHandler object.$",
                               Indexer,
                               filepath,
                               database=None,
                               filename_pattern="*.mseed")

    def test_download_leap_seconds_file(self):
        filepath = get_test_data_filepath()
        database = os.path.join(filepath, 'timeseries.sqlite')
        indexer = Indexer(filepath,
                          database=database)
        # mock actually downloading the file since this requires a internet
        # connection
        indexer._download = mock.MagicMock(return_value=requests.Response())
        # create a empty leap-seconds.list file
        test_file = os.path.join(
                            os.path.dirname(database), "leap-seconds.list")
        file_path = indexer.download_leap_seconds_file(test_file)
        # assert that the file was put in the same location as the
        # sqlite db
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, test_file)
        os.remove(test_file)

    def test_download_leap_seconds_file_no_path_given(self):
        filepath = get_test_data_filepath()
        database = os.path.join(filepath, 'timeseries.sqlite')
        indexer = Indexer(filepath,
                          database=database)
        # mock actually downloading the file since this requires a internet
        # connection
        indexer._download = mock.MagicMock(return_value=requests.Response())
        file_path = indexer.download_leap_seconds_file()

        self.assertEqual(
            os.path.normpath(file_path),
            os.path.normpath(os.path.join(os.path.dirname(database),
                                          "leap-seconds.list")))

        # assert that the file was put in the same location as the
        # sqlite db
        self.assertTrue(os.path.isfile(file_path))
        os.remove(file_path)

    def test__get_leap_seconds_file(self):
        filepath = get_test_data_filepath()
        database = os.path.join(filepath, 'timeseries.sqlite')
        indexer = Indexer(filepath,
                          database=database)

        # test that a bad leap second file path raises an error
        self.assertRaisesRegex(OSError,
                               "^No leap seconds file exists at.*$",
                               Indexer,
                               filepath,
                               database=database,
                               leap_seconds_file="/some/bad/path/")
        self.assertRaisesRegex(OSError,
                               "^No leap seconds file exists at.*$",
                               indexer._get_leap_seconds_file,
                               "/some/bad/path/")

        # test search
        # create a empty leap-seconds.list file
        test_file = os.path.normpath(os.path.join(
                            os.path.dirname(database), "leap-seconds.list"))
        open(test_file, 'a').close()
        file_path = os.path.normpath(indexer._get_leap_seconds_file("SEARCH"))
        self.assertEqual(file_path, test_file)
        os.remove(test_file)

    def test_build_file_list(self):
        filepath = get_test_data_filepath()
        database = os.path.join(filepath, 'timeseries.sqlite')
        indexer = Indexer(filepath,
                          database=database,
                          filename_pattern="*.mseed")

        # test for relative paths
        file_list = indexer.build_file_list(relative_paths=True,
                                            reindex=True)
        file_list.sort()
        self.assertEqual(len(file_list), 3)
        self.assertEqual(os.path.normpath(
            'CU/2018/001/CU.TGUH.00.BHZ.2018.001_first_minute.mseed'),
            file_list[0])
        self.assertEqual(os.path.normpath(
            'IU/2018/001/IU.ANMO.10.BHZ.2018.001_first_minute.mseed'),
            file_list[1])
        self.assertEqual(os.path.normpath(
            'IU/2018/001/IU.COLA.10.BHZ.2018.001_first_minute.mseed'),
            file_list[2])

        # case where the root path is outside of the absolute
        # data path, to assert that already indexed files are still skipped
        indexer = Indexer(tempfile.mkdtemp(),
                          database=TSIndexDatabaseHandler(database=database),
                          filename_pattern="*.mseed")
        self.assertRaisesRegex(OSError,
                               "^No files matching filename.*$",
                               indexer.build_file_list,
                               reindex=True)

        # test for absolute paths
        # this time pass a TSIndexDatabaseHandler instance as the database
        indexer = Indexer(filepath,
                          database=TSIndexDatabaseHandler(database=database),
                          filename_pattern="*.mseed",
                          leap_seconds_file=None)
        file_list = indexer.build_file_list(reindex=True)
        file_list.sort()
        self.assertEqual(len(file_list), 3)
        self.assertNotEqual(os.path.normpath(
            'CU/2018/001/CU.TGUH.00.BHZ.2018.001_first_minute.mseed'),
            file_list[0])
        self.assertIn(os.path.normpath(
            'CU/2018/001/CU.TGUH.00.BHZ.2018.001_first_minute.mseed'),
            file_list[0])
        self.assertNotEqual(os.path.normpath(
            'IU/2018/001/IU.ANMO.10.BHZ.2018.001_first_minute.mseed'),
            file_list[1])
        self.assertIn(os.path.normpath(
            'IU/2018/001/IU.ANMO.10.BHZ.2018.001_first_minute.mseed'),
            file_list[1])
        self.assertNotEqual(os.path.normpath(
            'IU/2018/001/IU.COLA.10.BHZ.2018.001_first_minute.mseed'),
            file_list[2])
        self.assertIn(os.path.normpath(
            'IU/2018/001/IU.COLA.10.BHZ.2018.001_first_minute.mseed'),
            file_list[2])
        # test that already indexed files (relative and absolute) get skipped.
        self.assertRaisesRegex(OSError,
                               "^No unindexed files matching filename.*$",
                               indexer.build_file_list,
                               reindex=False,
                               relative_paths=False)
        self.assertRaisesRegex(OSError,
                               "^No unindexed files matching filename.*$",
                               indexer.build_file_list,
                               reindex=False,
                               relative_paths=True)
        # for this test mock an unindexed file ('data.mseed') to ensure that
        # it gets added when reindex is True
        mocked_files = [
                'CU/2018/001/'
                'CU.TGUH.00.BHZ.2018.001_first_minute.mseed',
                'IU/2018/001/'
                'IU.ANMO.10.BHZ.2018.001_first_minute.mseed',
                'IU/2018/001/'
                'IU.COLA.10.BHZ.2018.001_first_minute.mseed',
                'data.mseed'
            ]
        for i in range(len(mocked_files)):
            mocked_files[i] = os.path.normpath(mocked_files[i])
        indexer._get_rootpath_files = mock.MagicMock(return_value=mocked_files)
        self.assertEqual(indexer.build_file_list(
                           reindex=False,
                           relative_paths=False),
                         ['data.mseed'])

    def test_run_bad_index_cmd(self):
        """
        Checks that an OSError is raised when there is an error running a
        index_cmd. (such as no command found.)
        """
        filepath = get_test_data_filepath()
        indexer = Indexer(filepath,
                          filename_pattern="*.mseed",
                          index_cmd="some_bad_command"
                          )

        self.assertRaisesRegex(OSError,
                               "^Required program.* is not installed.*$",
                               indexer.run)

    def test_run(self):
        my_uuid = uuid.uuid4().hex
        fname = 'test_timeseries_{}'.format(my_uuid)
        filepath = get_test_data_filepath()
        database = '{}{}.sqlite'.format(filepath, fname)
        try:
            indexer = Indexer(filepath,
                              database=database,
                              filename_pattern="*.mseed",
                              parallel=2)
            if indexer._is_index_cmd_installed():
                indexer.run(relative_paths=True)
                keys = ['network', 'station', 'location', 'channel',
                        'quality', 'starttime', 'endtime', 'samplerate',
                        'filename', 'byteoffset', 'bytes', 'hash',
                        'timeindex', 'timespans', 'timerates', 'format']
                NamedRow = namedtuple('NamedRow',
                                      keys)

                expected_tsindex_data = \
                    [
                     NamedRow(
                        "CU", "TGUH", "00", "BHZ", "M",
                        "2018-01-01T00:00:00.000000",
                        "2018-01-01T00:01:00.000000", 40.0,
                        "CU/2018/001/"
                        "CU.TGUH.00.BHZ.2018.001_first_minute.mseed",
                        0, 4096, "aaaac5315f84cdd174fd8360002a1e3a",
                        "1514764800.000000=>0,latest=>1",
                        "[1514764800.000000:1514764860.000000]", None, None),
                     NamedRow(
                        "IU", "ANMO", "10", "BHZ", "M",
                        "2018-01-01T00:00:00.019500",
                        "2018-01-01T00:00:59.994536", 40.0,
                        "IU/2018/001/"
                        "IU.ANMO.10.BHZ.2018.001_first_minute.mseed",
                        0, 2560, "36a771ca1dc648c505873c164d8b26f2",
                        "1514764800.019500=>0,latest=>1",
                        "[1514764800.019500:1514764859.994536]", None, None),
                     NamedRow(
                        "IU", "COLA", "10", "BHZ", "M",
                        "2018-01-01T00:00:00.019500",
                        "2018-01-01T00:00:59.994538", 40.0,
                        "IU/2018/001/"
                        "IU.COLA.10.BHZ.2018.001_first_minute.mseed",
                        0, 5120, "4ccbb97573ca00ef8c2c4f9c01d27ddf",
                        "1514764800.019500=>0,latest=>1",
                        "[1514764800.019500:1514764859.994538]", None, None)]
                db_handler = TSIndexDatabaseHandler(database=database)
                tsindex_data = db_handler._fetch_index_rows([("I*,C*", "*",
                                                              "0?,1?", "*",
                                                              "2018-01-01",
                                                              "2018-02-01")])

                for i in range(0, len(expected_tsindex_data)):
                    for j in range(0, len(keys)):
                        self.assertEqual(getattr(expected_tsindex_data[i],
                                                 keys[j]),
                                         getattr(tsindex_data[i], keys[j]))
                self.assertEqual(len(tsindex_data), len(expected_tsindex_data))
        finally:
            purge(filepath, '^{}.*$'.format(fname))


class TSIndexDatabaseHandlerTestCase(TestCase):

    def test_bad_sqlitdb_filepath(self):
        """
        Checks that an error is raised when an invalid path is provided to
        a SQLite database
        """
        filepath = get_test_data_filepath()
        self.assertRaisesRegex(OSError,
                               "^Database path.*does not exist.$",
                               Indexer,
                               filepath,
                               database='/some/bad/path/',
                               filename_pattern="*.mseed",
                               parallel=2)

    def test__fetch_summary_rows(self):
        # test with actual sqlite3 database that is missing a summary table
        # a temporary summary table gets created at runtime
        filepath = get_test_data_filepath()
        db_path = os.path.join(filepath, 'timeseries.sqlite')
        request_handler = TSIndexDatabaseHandler(db_path)

        keys = ['network', 'station', 'location', 'channel',
                'earliest', 'latest']
        NamedRow = namedtuple('NamedRow',
                              keys)

        expected_ts_summary_data = \
            [NamedRow(
                "CU", "TGUH", "00", "BHZ",
                "2018-01-01T00:00:00.000000",
                "2018-01-01T00:01:00.000000"),
             NamedRow(
                "IU", "ANMO", "10", "BHZ",
                "2018-01-01T00:00:00.019500",
                "2018-01-01T00:00:59.994536")]

        ts_summary_data = request_handler._fetch_summary_rows(
                                              [("I*,CU",
                                                "ANMO,T*",
                                                "00,10",
                                                "BHZ",
                                                "2018-01-01T00:00:00.000000",
                                                "2018-12-31T00:00:00.000000")])

        for i in range(0, len(expected_ts_summary_data)):
            for j in range(0, len(keys)):
                self.assertEqual(getattr(expected_ts_summary_data[i], keys[j]),
                                 getattr(ts_summary_data[i], keys[j]))

        # test for case where query returns no results
        ts_summary_data = request_handler._fetch_summary_rows(
                                              [("XX",
                                                "ANMO,T*",
                                                "00,10",
                                                "BHZ",
                                                "2018-01-01T00:00:00.000000",
                                                "2018-12-31T00:00:00.000000")])
        self.assertEqual(ts_summary_data, [])

    def test_get_tsindex_summary_cte(self):
        # test with actual sqlite3 database that is missing a summary table
        # a tsindex summary CTE gets created using the tsindex at runtime
        filepath = get_test_data_filepath()
        db_path = os.path.join(filepath, 'timeseries.sqlite')
        # supply an existing session
        engine = sa.create_engine("sqlite:///{}".format(db_path))
        session = sessionmaker(bind=engine)
        request_handler = TSIndexDatabaseHandler(session=session)

        ts_summary_cte = request_handler.get_tsindex_summary_cte()

        expected_ts_summary_data = \
            [("CU", "TGUH", "00", "BHZ",
              "2018-01-01T00:00:00.000000",
              "2018-01-01T00:01:00.000000"),
             ("IU", "ANMO", "10", "BHZ",
              "2018-01-01T00:00:00.019500",
              "2018-01-01T00:00:59.994536"),
             ("IU", "COLA", "10", "BHZ",
              "2018-01-01T00:00:00.019500",
              "2018-01-01T00:00:59.994538")]
        query_results = (session().query(ts_summary_cte))
        for idx, r in enumerate(query_results):
            result = r[:6]  # ignore updt date
            self.assertEqual(result, expected_ts_summary_data[idx])


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.makeSuite(ClientTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(IndexerTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(TSIndexDatabaseHandlerTestCase,
                                         'test'))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
