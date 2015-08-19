#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.download_helpers test suite.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2105
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import copy
import os
import shutil
import tempfile
import unittest

import numpy as np

import obspy
from obspy.core.compatibility import mock
from obspy.clients.fdsn.download_helpers import (domain, Restrictions,
                                                 DownloadHelper)
from obspy.clients.fdsn.download_helpers.utils import (
    filter_channel_priority, get_stationxml_filename, get_mseed_filename,
    get_stationxml_contents, SphericalNearestNeighbour, safe_delete,
    download_stationxml, download_and_split_mseed_bulk)
from obspy.clients.fdsn.download_helpers.download_status import (
    Channel, TimeInterval, Station, STATUS, ClientDownloadHelper)


class DomainTestCase(unittest.TestCase):
    """
    Test case for the domain definitions.
    """
    def test_rectangular_domain(self):
        """
        Test the rectangular domain.
        """
        dom = domain.RectangularDomain(-10, 10, -20, 20)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "minlatitude": -10,
            "maxlatitude": 10,
            "minlongitude": -20,
            "maxlongitude": 20})

        # The rectangular domain is completely defined by the query parameters.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_circular_domain(self):
        """
        Test the circular domain.
        """
        dom = domain.CircularDomain(10, 20, 30, 40)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "latitude": 10,
            "longitude": 20,
            "minradius": 30,
            "maxradius": 40})

        # The circular domain is completely defined by the query parameters.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_global_domain(self):
        """
        Test the global domain.
        """
        dom = domain.GlobalDomain()
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {})

        # Obviously every point is in the domain.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_subclassing_without_abstract_method(self):
        """
        Subclassing without implementing the get_query_parameters method
        results in a TypeError upon instantiation time.
        """
        class NewDom(domain.Domain):
            pass

        self.assertRaises(TypeError, NewDom)

    def test_instantiating_root_domain_object_fails(self):
        """
        Trying to create a root domain object should fail.
        """
        self.assertRaises(TypeError, domain.Domain)


class RestrictionsTestCase(unittest.TestCase):
    """
    Test case for the restrictions object.
    """
    def __init__(self, *args, **kwargs):
        super(RestrictionsTestCase, self).__init__(*args, **kwargs)
        self.path = os.path.dirname(__file__)
        self.data = os.path.join(self.path, "data")

    def test_restrictions_object(self):
        """
        Tests the restrictions object.
        """
        start = obspy.UTCDateTime(2014, 1, 1)
        res = Restrictions(starttime=start, endtime=start + 10)

        # No chunklength means it should just return one item.
        chunks = list(res)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], (start, start + 10))

        # One with chunklength should return the chunked pieces.
        res = Restrictions(starttime=start, endtime=start + 10,
                           chunklength_in_sec=1)
        chunks = list(res)
        self.assertEqual(len(chunks), 10)
        self.assertEqual(
            [_i[0] for _i in chunks],
            [start + _i * 1 for _i in range(10)])
        self.assertEqual(
            [_i[1] for _i in chunks],
            [start + _i * 1 for _i in range(1, 11)])
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], start + 10)

        # Make sure the last piece is cut if it needs to be.
        start = obspy.UTCDateTime(2012, 1, 1)
        end = obspy.UTCDateTime(2012, 2, 1)
        res = Restrictions(starttime=start, endtime=end,
                           chunklength_in_sec=86400 * 10)
        chunks = list(res)
        self.assertEqual(chunks, [
            (start, start + 86400 * 10),
            (start + 86400 * 10, start + 86400 * 20),
            (start + 86400 * 20, start + 86400 * 30),
            (start + 86400 * 30, end)])

        # No station start-and endtime by default
        res = Restrictions(starttime=start, endtime=start + 10)
        self.assertEqual(res.station_starttime, None)
        self.assertEqual(res.station_endtime, None)

        # One can only set one of the two.
        res = Restrictions(starttime=start, endtime=start + 10,
                           station_starttime=start - 10)
        self.assertEqual(res.station_starttime, start - 10)
        self.assertEqual(res.station_endtime, None)

        res = Restrictions(starttime=start, endtime=start + 10,
                           station_endtime=start + 20)
        self.assertEqual(res.station_starttime, None)
        self.assertEqual(res.station_endtime, start + 20)

        # Will raise a ValueError if either within the time interval of the
        # normal start- and endtime.
        self.assertRaises(ValueError, Restrictions, starttime=start,
                          endtime=start+10, station_starttime=start + 1)

        self.assertRaises(ValueError, Restrictions, starttime=start,
                          endtime=start+10, station_endtime=start + 9)

        # Fine if they are equal with both.
        Restrictions(starttime=start, endtime=start + 10,
                     station_starttime=start, station_endtime=start + 10)



class DownloadHelpersUtilTestCase(unittest.TestCase):
    """
    Test cases for utility functionality for the download helpers.
    """
    def __init__(self, *args, **kwargs):
        super(DownloadHelpersUtilTestCase, self).__init__(*args, **kwargs)
        self.path = os.path.dirname(__file__)
        self.data = os.path.join(self.path, "data")

    def test_channel_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60)
            for _i in range(10)]
        c1 = Channel("", "BHE", time_intervals)
        c2 = Channel("10", "SHE", time_intervals)
        c3 = Channel("00", "BHZ", time_intervals)
        c4 = Channel("", "HHE", time_intervals)
        channels = [c1, c2, c3, c4]

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "HH[Z,N,E]", "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]",
                "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c4])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c1, c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH*", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c1, c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH[N,Z]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["S*", "BH*"])
        self.assertEqual(filtered_channels, [c2])

        # Different ways to not filter.
        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=None)
        self.assertEqual(filtered_channels, channels)

    def test_location_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60)
            for _i in range(10)]
        c1 = Channel("", "BHE", time_intervals)
        c2 = Channel("10", "SHE", time_intervals)
        c3 = Channel("00", "BHZ", time_intervals)
        c4 = Channel("", "HHE", time_intervals)
        channels = [c1, c2, c3, c4]

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*0"])
        self.assertEqual(filtered_channels, [c2, c3])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["00"])
        self.assertEqual(filtered_channels, [c3])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=[""])
        self.assertEqual(filtered_channels, [c1, c4])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["1?"])
        self.assertEqual(filtered_channels, [c2])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["", "*0"])
        self.assertEqual(filtered_channels, [c1, c4])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*0", ""])
        self.assertEqual(filtered_channels, [c2, c3])

        # Different ways to not filter.
        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=None)
        self.assertEqual(filtered_channels, channels)

    def test_spherical_nearest_neighbour(self):
        """
        Tests the spherical kd-tree.
        """
        # Get the distance of a point to itself.
        point = Station("", "", 10.0, 20.0, [])
        tree = SphericalNearestNeighbour(data=[point])
        result = tree.query(points=[point])
        distance, indices = result[0][0], result[1][0]
        np.testing.assert_allclose(distance, [0])
        np.testing.assert_allclose(indices, [0])

        # Two points, one 50 km, the other 150 km distant.
        point_a = Station("", "", 0.0, -0.5, [])
        point_b = Station("", "", 0.0, 1.5, [])
        tree = SphericalNearestNeighbour(data=[point_a, point_b])
        result = tree.query(points=[Station("", "", 0.0, 0.0, [])])
        distance, indices = result[0][0], result[1][0]

        np.testing.assert_allclose(distance, [55597.36, 166787.862],
                                   atol=1, rtol=1E-5)
        np.testing.assert_allclose(indices, [0, 1])

        # Query pairs.
        point_a = Station("", "", 0.0, -0.5, [])
        point_b = Station("", "", 0.0, 1.5, [])
        point_c = Station("", "", 0.0, 0.0, [])
        tree = SphericalNearestNeighbour(data=[point_a, point_b, point_c])
        # 100 km apart. Only contains points a and c.
        self.assertEqual(tree.query_pairs(100000), {(0, 2)})

    def test_safe_delete(self):
        """
        Test the safe-delete function.
        """
        dir = tempfile.mkdtemp()
        try:
            # If the file does not exist, nothing happens.
            safe_delete(os.path.join(dir, "non-existant"))
            # If not a file, an error will be raised.
            name = os.path.join(dir, "tmpdir")
            os.makedirs(name)
            self.assertRaises(ValueError, safe_delete, name)
            # Otherwise it can delete a file just fine.
            name = os.path.join(dir, "tmpfile")
            with open(name, "wt") as fh:
                fh.write("0")
            self.assertTrue(os.path.exists(name))
            safe_delete(name)
            self.assertFalse(os.path.exists(name))

        finally:
            shutil.rmtree(dir)

    def test_download_stationxml(self):
        """
        Mock test for the StationXML downloading.

        Does not do much and is not a proper test but its something and
        makes sure there is not obvious logic error.
        """
        bulk = [
            ["BW", "ALTM"],
            ["BW", "ALTM"],
        ]
        filename = "temp.xml"
        client_name = "mock"

        client = mock.MagicMock()
        logger = mock.MagicMock()

        # Normal call.
        ret_val = download_stationxml(client, client_name, bulk, filename,
                                      logger)
        self.assertEqual(ret_val, (("BW", "ALTM"), filename))

        self.assertEqual(logger.info.call_count, 1)
        self.assertEqual(logger.info.call_args[0][0],
                         "Client 'mock' - Successfully downloaded 'temp.xml'.")
        self.assertEqual(client.get_stations_bulk.call_count, 1)
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["bulk"], bulk)
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["level"], "response")
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["filename"], filename)

        # Call that raises.
        client.reset_mock()
        logger.reset_mock()

        def raise_exception():
            raise ValueError("Test")
        client.get_stations_bulk.side_effect = raise_exception

        ret_val = download_stationxml(client, client_name, bulk, filename,
                                      logger)
        self.assertEqual(ret_val, None)

        self.assertEqual(logger.info.call_count, 1)
        self.assertEqual(
            logger.info.call_args[0][0],
            "Failed to download StationXML from 'mock' for station 'BW.ALTM'.")
        self.assertEqual(client.get_stations_bulk.call_count, 1)
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["bulk"], bulk)
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["level"], "response")
        self.assertEqual(
            client.get_stations_bulk.call_args[1]["filename"], filename)

    def test_download_and_split_mseed(self):
        """
        Largely mocked test for the download_and_split_mseed() function.
        """
        client_name = "mock"

        client = mock.MagicMock()
        logger = mock.MagicMock()

        def get_waveforms_bulk_mock(bulk, filename):
            """
            Actually create the requested MiniSEED file.
            """
            st = obspy.Stream()
            for item in bulk:
                tr = obspy.Trace()
                tr.stats.network = item[0]
                tr.stats.station = item[1]
                tr.stats.location = item[2]
                tr.stats.channel = item[3]
                tr.stats.starttime = item[4]
                tr.stats.delta = 1.0
                tr.data = np.empty(int(item[5] - item[4]) + 1)
                st.traces.append(tr)

            st.write(filename, format="mseed")

        client.get_waveforms_bulk.side_effect = get_waveforms_bulk_mock

        tmpdir = tempfile.mkdtemp()

        try:
            chunks = [
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_1.mseed")],
                ["BW", "ALTM", "", "EHN", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_2.mseed")],
                ["BW", "ALTM", "", "EHZ", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_3.mseed")],
            ]
            ret_val = download_and_split_mseed_bulk(
                client=client, client_name=client_name, chunks=chunks,
                logger=logger)

            contents = [("file_1.mseed", "BW.ALTM..EHE"),
                        ("file_2.mseed", "BW.ALTM..EHN"),
                        ("file_3.mseed", "BW.ALTM..EHZ")]

            self.assertEqual(ret_val,
                             sorted([os.path.join(tmpdir, _i[0])
                                     for _i in contents]))

            # Make sure all files have been written.
            self.assertEqual(sorted(os.listdir(tmpdir)),
                             ["file_1.mseed", "file_2.mseed", "file_3.mseed"])
            # Check the actual files.
            for filename, id, in contents:
                st = obspy.read(os.path.join(tmpdir, filename))
                self.assertEqual(len(st), 1)
                tr = st[0]
                self.assertEqual(tr.id, id)
                self.assertEqual(tr.stats.starttime, obspy.UTCDateTime(0))
                self.assertEqual(tr.stats.endtime, obspy.UTCDateTime(10))

        finally:
            shutil.rmtree(tmpdir)

        # Same as before but now add some random other things so make sure
        # they get filtered out.
        client.reset_mock()
        logger.reset_mock()

        def get_waveforms_bulk_mock(bulk, filename):
            """
            Actually create the requested MiniSEED file.
            """
            st = obspy.Stream()
            for item in bulk:
                tr = obspy.Trace()
                tr.stats.network = item[0]
                tr.stats.station = item[1]
                tr.stats.location = item[2]
                tr.stats.channel = item[3]
                tr.stats.starttime = item[4]
                tr.stats.delta = 1.0
                tr.data = np.empty(int(item[5] - item[4]) + 1)
                st.traces.append(tr)

            # Add some random other stuff to mess with things.
            tr = obspy.Trace()
            tr.stats.network = "HM"
            tr.stats.channel = "EHE"
            tr.data = np.empty(12)
            st.traces.append(tr)

            tr = obspy.Trace()
            tr.stats.network = "HM"
            tr.stats.channel = "BHE"
            tr.data = np.empty(12)
            st.traces.append(tr)

            # This time same id as above, but different time span.
            tr = obspy.Trace()
            tr.stats.network = bulk[0][0]
            tr.stats.station = bulk[0][1]
            tr.stats.location = bulk[0][2]
            tr.stats.channel = bulk[0][3]
            tr.data = np.empty(34)
            tr.stats.starttime += 1234567.345
            st.traces.append(tr)

            st.write(filename, format="mseed")

        client.get_waveforms_bulk.side_effect = get_waveforms_bulk_mock

        tmpdir = tempfile.mkdtemp()

        try:
            chunks = [
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_1.mseed")],
                ["BW", "ALTM", "", "EHN", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_2.mseed")],
                ["BW", "ALTM", "", "EHZ", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(10), os.path.join(tmpdir, "file_3.mseed")],
            ]
            ret_val = download_and_split_mseed_bulk(
                client=client, client_name=client_name, chunks=chunks,
                logger=logger)

            contents = [("file_1.mseed", "BW.ALTM..EHE"),
                        ("file_2.mseed", "BW.ALTM..EHN"),
                        ("file_3.mseed", "BW.ALTM..EHZ")]

            self.assertEqual(ret_val,
                             sorted([os.path.join(tmpdir, _i[0])
                                     for _i in contents]))

            # Make sure all files have been written.
            self.assertEqual(sorted(os.listdir(tmpdir)),
                             ["file_1.mseed", "file_2.mseed", "file_3.mseed"])
            # Check the actual files.
            for filename, id, in contents:
                st = obspy.read(os.path.join(tmpdir, filename))
                self.assertEqual(len(st), 1)
                tr = st[0]
                self.assertEqual(tr.id, id)
                self.assertEqual(tr.stats.starttime, obspy.UTCDateTime(0))
                self.assertEqual(tr.stats.endtime, obspy.UTCDateTime(10))

        finally:
            shutil.rmtree(tmpdir)

        # Now simulate a request of lots of data from the same channel.
        client.reset_mock()
        logger.reset_mock()

        def get_waveforms_bulk_mock(bulk, filename):
            """
            Actually create the requested MiniSEED file.
            """
            st = obspy.Stream()
            for item in bulk:
                tr = obspy.Trace()
                tr.stats.network = item[0]
                tr.stats.station = item[1]
                tr.stats.location = item[2]
                tr.stats.channel = item[3]
                tr.stats.starttime = item[4]
                tr.stats.delta = 1.0
                tr.data = np.empty(int(item[5] - item[4]) + 1)
                st.traces.append(tr)

            st.write(filename, format="mseed")

        client.get_waveforms_bulk.side_effect = get_waveforms_bulk_mock

        tmpdir = tempfile.mkdtemp()

        try:
            chunks = [
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(1E5), os.path.join(tmpdir, "file_1.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(1E5),
                 obspy.UTCDateTime(2E5), os.path.join(tmpdir, "file_2.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(2E5),
                 obspy.UTCDateTime(3E5), os.path.join(tmpdir, "file_3.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(3E5),
                 obspy.UTCDateTime(4E5), os.path.join(tmpdir, "file_4.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(4E5),
                 obspy.UTCDateTime(5E5), os.path.join(tmpdir, "file_5.mseed")],
                 ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(6E5),
                  obspy.UTCDateTime(7E5), os.path.join(tmpdir, "file_6.mseed")]
            ]
            ret_val = download_and_split_mseed_bulk(
                client=client, client_name=client_name, chunks=chunks,
                logger=logger)

            # Nof five files but all the same channel.
            contents = [("file_1.mseed", "BW.ALTM..EHE"),
                        ("file_2.mseed", "BW.ALTM..EHE"),
                        ("file_3.mseed", "BW.ALTM..EHE"),
                        ("file_4.mseed", "BW.ALTM..EHE"),
                        ("file_5.mseed", "BW.ALTM..EHE"),
                        ("file_6.mseed", "BW.ALTM..EHE")]

            self.assertEqual(ret_val,
                             sorted([os.path.join(tmpdir, _i[0])
                                     for _i in contents]))

            # Make sure all files have been written.
            self.assertEqual(sorted(os.listdir(tmpdir)),
                             [_i[0] for _i in contents])

            # The interesting thing here is that it should only send a
            # request for single time span and then split again on the
            # client side. Here is two time spans as one segment is further
            # away.
            call_args = client.get_waveforms_bulk.call_args[0][0]
            self.assertEqual(
                call_args,[
                    ['BW', 'ALTM', '', 'EHE', obspy.UTCDateTime(0),
                     obspy.UTCDateTime(5E5)],
                    ['BW', 'ALTM', '', 'EHE', obspy.UTCDateTime(6E5),
                     obspy.UTCDateTime(7E5)]])

        finally:
            shutil.rmtree(tmpdir)

        # Last one attempting to get overlapping filenames.
        client.reset_mock()
        logger.reset_mock()

        def get_waveforms_bulk_mock(bulk, filename):
            """
            Actually create the requested MiniSEED file.
            """
            st = obspy.Stream()
            for item in bulk:
                tr = obspy.Trace()
                tr.stats.network = item[0]
                tr.stats.station = item[1]
                tr.stats.location = item[2]
                tr.stats.channel = item[3]
                tr.stats.starttime = item[4]
                tr.stats.delta = 1.0
                tr.data = np.empty(int(item[5] - item[4]) + 1)
                st.traces.append(tr)

            st.write(filename, format="mseed")

        client.get_waveforms_bulk.side_effect = get_waveforms_bulk_mock

        tmpdir = tempfile.mkdtemp()

        try:
            chunks = [
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(0),
                 obspy.UTCDateTime(1E5), os.path.join(tmpdir, "file_1.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(0.4E5),
                 obspy.UTCDateTime(1.6E5),
                 os.path.join(tmpdir, "file_2.mseed")],
                ["BW", "ALTM", "", "EHE", obspy.UTCDateTime(1.2E5),
                 obspy.UTCDateTime(2.2E5),
                 os.path.join(tmpdir, "file_3.mseed")]]
            ret_val = download_and_split_mseed_bulk(
                client=client, client_name=client_name, chunks=chunks,
                logger=logger)

            contents = [("file_1.mseed", "BW.ALTM..EHE"),
                        ("file_2.mseed", "BW.ALTM..EHE"),
                        ("file_3.mseed", "BW.ALTM..EHE")]

            self.assertEqual(ret_val,
                             sorted([os.path.join(tmpdir, _i[0])
                                     for _i in contents]))

            # Make sure all files have been written.
            self.assertEqual(sorted(os.listdir(tmpdir)),
                             ["file_1.mseed", "file_2.mseed", "file_3.mseed"])
            # Check the actual files. There will be no overlap of data in
            # the files but the data should be distributed across
            # files according to some heuristics.
            st = obspy.read(os.path.join(tmpdir, "file_1.mseed"))
            self.assertEqual(len(st), 1)
            tr = st[0]
            self.assertEqual(tr.id, "BW.ALTM..EHE")
            self.assertEqual(tr.stats.starttime, obspy.UTCDateTime(0))
            # Record length of 512.
            self.assertTrue(
                abs(tr.stats.endtime - obspy.UTCDateTime(1E5)) < 512)

            st = obspy.read(os.path.join(tmpdir, "file_2.mseed"))
            self.assertEqual(len(st), 1)
            tr = st[0]
            self.assertEqual(tr.id, "BW.ALTM..EHE")
            # Record length of 512.
            self.assertTrue(
                abs(tr.stats.starttime - obspy.UTCDateTime(1E5)) < 512)
            self.assertTrue(
                abs(tr.stats.endtime - obspy.UTCDateTime(1.6E5)) < 512)

            st = obspy.read(os.path.join(tmpdir, "file_3.mseed"))
            self.assertEqual(len(st), 1)
            tr = st[0]
            self.assertEqual(tr.id, "BW.ALTM..EHE")
            # Record length of 512.
            self.assertTrue(
                abs(tr.stats.starttime - obspy.UTCDateTime(1.6E5)) < 512)
            # Endtime is exact again as no more overlaps occur.
            self.assertEqual(tr.stats.endtime, obspy.UTCDateTime(2.2E5))

        finally:
            shutil.rmtree(tmpdir)


    # def test_merge_station_lists(self):
    #     """
    #     Tests the merging of two stations.
    #     """
    #     list_one = [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("11", "11", 0, 0, 500, [], None),
    #         Station("11", "11", 0, 0, 1500, [], None),
    #     ]
    #     list_two = [
    #         Station("11", "11", 0, 0, 10, [], None),
    #         Station("11", "11", 0, 0, 505, [], None),
    #         Station("11", "11", 0, 0, 1505, [], None),
    #     ]
    #     new_list = filter_based_on_interstation_distance(list_one, list_two,
    #                                                      20)
    #     self.assertEqual(new_list, list_one)
    #     new_list = filter_based_on_interstation_distance(list_one, list_two, 2)
    #     self.assertEqual(new_list, list_one + list_two)
    #     new_list = filter_based_on_interstation_distance(list_one, list_two, 8)
    #     self.assertEqual(new_list, list_one + [
    #         Station("11", "11", 0, 0, 10, [], None)])

    def test_stationxml_filename_helper(self):
        """
        Tests the get_stationxml_filename() function.
        """
        c1 = ("", "BHE")
        c2 = ("10", "SHE")
        starttime = obspy.UTCDateTime(2012, 1, 1)
        endtime = obspy.UTCDateTime(2012, 1, 2)
        channels = [c1, c2]

        # A normal string is considered a path.
        self.assertEqual(get_stationxml_filename(
            "FOLDER", network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            os.path.join("FOLDER", "BW.FURT.xml"))
        self.assertEqual(get_stationxml_filename(
            "stations", network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            os.path.join("stations", "BW.FURT.xml"))

        # Passing a format string causes it to be used.
        self.assertEqual(get_stationxml_filename(
            "{network}_{station}.xml", network="BW", station="FURT",
            channels=channels, starttime=starttime, endtime=endtime),
            "BW_FURT.xml")
        self.assertEqual(get_stationxml_filename(
            "TEMP/{network}/{station}.xml", network="BW", station="FURT",
            channels=channels, starttime=starttime, endtime=endtime),
            "TEMP/BW/FURT.xml")

        # A passed function will be executed. A string should just be returned.
        def get_name(network, station, channels, starttime, endtime):
            return "network" + "__" + station
        self.assertEqual(get_stationxml_filename(
            get_name, network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime), "network__FURT")

        # A dictionary with certain keys are also acceptable.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2],
                    "filename": "test.xml"}
        self.assertEqual(get_stationxml_filename(
            get_name, network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            {"missing_channels": [c1], "available_channels": [c2],
             "filename": "test.xml"})

        # Missing keys raise.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2]}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        # Wrong value types should also raise.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2],
                    "filename": True}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": True,
                    "available_channels": [c2],
                    "filename": "test.xml"}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": True,
                    "filename": "test.xml"}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_stationxml_filename, lambda x: 1,
                          "BW", "FURT", starttime, endtime)

    def test_mseed_filename_helper(self):
        """
        Tests the get_mseed_filename() function.
        """
        starttime = obspy.UTCDateTime(2014, 1, 2, 3, 4, 5)
        endtime = obspy.UTCDateTime(2014, 2, 3, 4, 5, 6)

        # A normal string is considered a path.
        self.assertEqual(
            get_mseed_filename("FOLDER", network="BW", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime),
            os.path.join(
                "FOLDER", "BW.FURT..BHE__2014-01-02T03-04-05Z__"
                "2014-02-03T04-05-06Z.mseed"))
        self.assertEqual(
            get_mseed_filename("waveforms", network="BW", station="FURT",
                               location="00", channel="BHE",
                               starttime=starttime, endtime=endtime),
            os.path.join("waveforms", "BW.FURT.00.BHE__2014-01-02T03-04-05Z__"
                         "2014-02-03T04-05-06Z.mseed"))

        # Passing a format string causes it to be used.
        self.assertEqual(get_mseed_filename(
            "{network}_{station}_{location}_{channel}_"
            "{starttime}_{endtime}.ms", network="BW", station="FURT",
            location="", channel="BHE", starttime=starttime, endtime=endtime),
            "BW_FURT__BHE_2014-01-02T03-04-05Z_2014-02-03T04-05-06Z.ms")
        self.assertEqual(get_mseed_filename(
            "{network}_{station}_{location}_{channel}_"
            "{starttime}_{endtime}.ms", network="BW", station="FURT",
            location="00", channel="BHE", starttime=starttime,
            endtime=endtime),
            "BW_FURT_00_BHE_2014-01-02T03-04-05Z_2014-02-03T04-05-06Z.ms")

        # A passed function will be executed.
        def get_name(network, station, location, channel, starttime, endtime):
            if network == "AH":
                return True
            return "network" + "__" + station + location + channel

        # Returning a filename is possible.
        self.assertEqual(
            get_mseed_filename(get_name, network="BW", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime),
            "network__FURTBHE")
        # 'True' can also be returned. This indicates that the file already
        # exists.
        self.assertEqual(
            get_mseed_filename(get_name, network="AH", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime), True)

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_mseed_filename, lambda x: 1,
                          "BW", "FURT", "", "BHE")

    def test_get_stationxml_contents(self):
        """
        Tests the fast get_stationxml_contents() function.
        """
        ChannelAvailability = collections.namedtuple(
            "ChannelAvailability",
            ["network", "station", "location", "channel", "starttime",
             "endtime", "filename"])

        filename = os.path.join(self.data, "AU.MEEK.xml")
        # Read with ObsPy and the fast variant.
        inv = obspy.read_inventory(filename)
        # Consistency test.
        self.assertEqual(len(inv.networks), 1)
        contents = get_stationxml_contents(filename)
        net = inv[0]
        sta = net[0]
        cha = sta[0]
        self.assertEqual(
            contents,
            [ChannelAvailability(net.code, sta.code, cha.location_code,
                                 cha.code, cha.start_date, cha.end_date,
                                 filename)])

    def test_channel_str_representation(self):
        """
        Test the string representations of channel objects.
        """
        # Single interval.
        intervals = [TimeInterval(
            obspy.UTCDateTime(2012, 1, 1), obspy.UTCDateTime(2012, 2, 1),
            filename=None, status=None)]
        c = Channel(location="", channel="BHE", intervals=intervals)
        self.assertEqual(str(c), (
            "Channel '.BHE':\n"
            "\tTimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 2, 1, 0, 0), filename=None, "
            "status='none')"))


class TimeIntervalTestCase(unittest.TestCase):
    """
    Test cases for the TimeInterval class.
    """
    def test_repr(self):
        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et)
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename=None, status='none')")

        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et, filename="dummy.txt")
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename='dummy.txt', "
            "status='none')")

        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et, filename="dummy.txt", status=STATUS.IGNORE)
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename='dummy.txt', "
            "status='ignore')")


class ChannelTestCase(unittest.TestCase):
    """
    Test cases for the Channel class.
    """
    def test_temporal_bounds(self):
        """
        Tests the temporal bounds retrieval.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c = Channel(location="", channel="BHZ", intervals=time_intervals)
        self.assertEqual(c.temporal_bounds, (st, st + 10 * 60))

    def test_wants_station_information(self):
        """
        Tests the wants station information property.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c = Channel(location="", channel="BHZ", intervals=time_intervals)

        # Right now all intervals have status NONE.
        self.assertFalse(c.needs_station_file)

        # As soon as at least one interval has status DOWNLOADED or EXISTS,
        # a station file is required.
        c.intervals[1].status = STATUS.EXISTS
        self.assertTrue(c.needs_station_file)
        c.intervals[1].status = STATUS.DOWNLOADED
        self.assertTrue(c.needs_station_file)

        # Any other status does not trigger the need to download.
        c.intervals[1].status = STATUS.DOWNLOAD_REJECTED
        self.assertFalse(c.needs_station_file)


class StationTestCase(unittest.TestCase):
    """
    Test cases for the Station class.
    """
    def test_has_existing_or_downloaded_time_intervals(self):
        """
        Tests for the property testing for existing or downloaded time
        intervals.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.copy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.copy(time_intervals))
        channels = [c1, c2]

        # False per default.
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)
        self.assertFalse(station.has_existing_or_downloaded_time_intervals)

        # Changing one interval to DOWNLOADED affects the whole station.
        station.channels[0].intervals[0].status = STATUS.DOWNLOADED
        self.assertTrue(station.has_existing_or_downloaded_time_intervals)

        # Some with EXISTS.
        station.channels[0].intervals[0].status = STATUS.EXISTS
        self.assertTrue(station.has_existing_or_downloaded_time_intervals)

        # Changing back.
        station.channels[0].intervals[0].status = STATUS.NONE
        self.assertFalse(station.has_existing_or_downloaded_time_intervals)

    def test_has_existing_time_intervals(self):
        """
        Tests for the property testing for existing time intervals.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.copy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.copy(time_intervals))
        channels = [c1, c2]

        # False per default.
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)
        self.assertFalse(station.has_existing_time_intervals)

        # Changing one interval to DOWNLOADED does not do anything
        station.channels[0].intervals[0].status = STATUS.DOWNLOADED
        self.assertFalse(station.has_existing_time_intervals)

        # EXISTS on the other hand does.
        station.channels[0].intervals[0].status = STATUS.EXISTS
        self.assertTrue(station.has_existing_time_intervals)

        # Changing back.
        station.channels[0].intervals[0].status = STATUS.NONE
        self.assertFalse(station.has_existing_time_intervals)

    def test_remove_files(self):
        """
        The remove files function removes all files of the station that have
        been downloaded.

        This test mocks the os.path.exists() function to always return True
        and the utils.safe_delete() function to assure it has been called
        when appropriate.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.deepcopy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.deepcopy(time_intervals))
        channels = [c1, c2]
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)

        logger = mock.MagicMock()

        with mock.patch("os.path.exists") as exists_mock:
            exists_mock.return_value = True

            with mock.patch("obspy.clients.fdsn.download_helpers"
                            ".utils.safe_delete") as p:
                # All status are NONE thus nothing should be deleted.
                station.remove_files(logger)
                self.assertEqual(p.call_count, 0)
                self.assertEqual(exists_mock.call_count, 0)

                # Set a random filename.
                filename = "/tmp/random.xml"
                station.stationxml_filename = filename
                # The setter of the stationxml_filename attribute should
                # check if the directory of the file already exists.
                self.assertEqual(exists_mock.call_count, 1)
                exists_mock.reset_mock()

                # Set the status of the file to DOWNLOADED. It should now be
                # downloaded.
                station.stationxml_status = STATUS.DOWNLOADED
                station.remove_files(logger)
                self.assertEqual(p.call_count, 1)
                self.assertEqual(p.call_args[0][0], filename)
                self.assertEqual(exists_mock.call_args[0][0], filename)
                exists_mock.reset_mock()
                p.reset_mock()

                # Now do the same with one of the time intervals.
                filename = "/tmp/random.mseed"
                station.stationxml_status = STATUS.NONE
                c1.intervals[0].filename = filename
                c1.intervals[0].status = STATUS.DOWNLOADED
                station.remove_files(logger)
                self.assertEqual(exists_mock.call_count, 1)
                self.assertEqual(p.call_count, 1)
                self.assertEqual(p.call_args[0][0], filename)
                self.assertEqual(exists_mock.call_args[0][0], filename)

    def test_temporal_bounds(self):
        """
        Tests the temporal bounds property.
        :return:
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.deepcopy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.deepcopy(time_intervals))
        channels = [c1, c2]
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)

        self.assertEqual(station.temporal_bounds, (st, st + 10 * 60))

    def test_sanitize_downloads(self):
        """
        Tests the sanitize_downloads() methods.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.deepcopy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.deepcopy(time_intervals))
        channels = [c1, c2]
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)

        logger = mock.MagicMock()

        with mock.patch("obspy.clients.fdsn.download_helpers"
                        ".utils.safe_delete") as p:
            # By default, nothing will happen.
            station.sanitize_downloads(logger)
            self.assertEqual(p.call_count, 0)
            p.reset_mock()

            # The whole purpose of the method is to make sure that each
            # MiniSEED files has a corresponding StationXML file. MiniSEED
            # files that do not fulfill this requirement will be deleted.
            # Fake it.
            filename = "tmp/file.mseed"
            c1.intervals[0].status = STATUS.DOWNLOADED
            c1.intervals[0].filename = filename
            c1.intervals[1].status = STATUS.DOWNLOADED
            c1.intervals[1].filename = filename

            # Right no channel has been marked missing, thus nothing should
            # happen.
            station.sanitize_downloads(logger)
            self.assertEqual(p.call_count, 0)
            p.reset_mock()

            # Mark one as missing and the corresponding information should
            # be deleted
            station.miss_station_information[("", "BHZ")] = True
            station.sanitize_downloads(logger)
            self.assertEqual(p.call_count, 2)
            # The status of the channel should be adjusted
            self.assertEqual(c1.intervals[0].status, STATUS.DOWNLOAD_REJECTED)
            self.assertEqual(c1.intervals[1].status, STATUS.DOWNLOAD_REJECTED)
            p.reset_mock()

    def test_prepare_mseed_download(self):
        """
        Tests the prepare_mseed download method.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c1 = Channel(location="", channel="BHZ",
                     intervals=copy.deepcopy(time_intervals))
        c2 = Channel(location="00", channel="EHE",
                     intervals=copy.deepcopy(time_intervals))
        channels = [c1, c2]
        all_tis = []
        for chan in channels:
            all_tis.extend(chan.intervals)
        station = Station(network="TA", station="A001", latitude=1,
                          longitude=2, channels=channels)

        # Patch the os.makedirs() function as it might be called at various
        # times.
        with mock.patch("os.makedirs") as p:
            # Now the output strongly depends on the `mseed_storage` keyword
            # argument. If it always returns True, all channels should be
            # ignored.
            station.prepare_mseed_download(
                mseed_storage=lambda *args, **kwargs: True)
            self.assertEqual(p.call_count, 0)
            for i in all_tis:
                self.assertEqual(i.status, STATUS.IGNORE)
            p.reset_mock()

            # Now if we just pass a string, it will be interpreted as a
            # foldername. It will naturally not exist, so all time intervals
            # will be marked as needing a download.
            with mock.patch("os.path.exists") as p_ex:
                p_ex.return_value = False
                station.prepare_mseed_download(
                    mseed_storage="/some/super/random/FJSD34J0J/path")
            # There are 20 time intervals.
            self.assertEqual(p.call_count, 20)
            # Once for each file, once for each folder.
            self.assertEqual(p_ex.call_count, 40)
            p.reset_mock()
            for i in all_tis:
                self.assertEqual(i.status, STATUS.NEEDS_DOWNLOADING)

            # Last but not least, if the files already exist, they will be
            # marked as that.
            with mock.patch("os.path.exists") as p_ex:
                p_ex.return_value = True
                station.prepare_mseed_download(
                    mseed_storage="/some/super/random/FJSD34J0J/path")
            # No folder should be created.
            self.assertEqual(p.call_count, 0)
            # Once for each file.
            self.assertEqual(p_ex.call_count, 20)
            p.reset_mock()
            for i in all_tis:
                self.assertEqual(i.status, STATUS.EXISTS)

    def test_prepare_stationxml_download_simple_cases(self):
        """
        Tests the simple cases of the prepare_stationxml_download() method.
        This method is crucial for everything to work thus it is tested
        rather extensively.
        """
        logger = mock.MagicMock()

        def _create_station():
            st = obspy.UTCDateTime(2015, 1, 1)
            time_intervals = [
                TimeInterval(st + _i * 60, st + (_i + 1) * 60)
                for _i in range(10)]
            c1 = Channel(location="", channel="BHZ",
                         intervals=copy.deepcopy(time_intervals))
            c2 = Channel(location="00", channel="EHE",
                         intervals=copy.deepcopy(time_intervals))
            channels = [c1, c2]
            all_tis = []
            for chan in channels:
                all_tis.extend(chan.intervals)
            station = Station(network="TA", station="A001", latitude=1,
                              longitude=2, channels=channels)
            return station

        temporal_bounds = (obspy.UTCDateTime(2015, 1, 1),
                           obspy.UTCDateTime(2015, 1, 1, 0, 10))

        # Again mock the os.makedirs function as it is called quite a bit.
        with mock.patch("os.makedirs") as p:

            # No time interval has any data, thus no interval actually needs
            # a station file. An interval only needs a station file if it
            # either downloaded data or if the data already exists.
            station = _create_station()
            station.prepare_stationxml_download(stationxml_storage="random",
                                                logger=logger)
            self.assertEqual(p.call_count, 0)
            self.assertEqual(station.stationxml_status, STATUS.NONE)
            self.assertEqual(station.want_station_information, {})
            self.assertEqual(station.miss_station_information, {})
            self.assertEqual(station.have_station_information, {})
            p.reset_mock()

            # Now the get_stationxml_filename() function will return a
            # string and the filename does not yet exists. All time
            # intervals require station information.
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            with mock.patch("os.path.exists") as exists_p:
                exists_p.return_value = False
                station.prepare_stationxml_download(
                    stationxml_storage="random", logger=logger)
            # Called twice, once for the directory, once for the file. Both
            # return False as enforced with the mock.
            self.assertEqual(exists_p.call_count, 2)
            self.assertEqual(exists_p.call_args_list[0][0][0], "random")
            self.assertEqual(exists_p.call_args_list[1][0][0],
                             os.path.join("random", "TA.A001.xml"))
            # Thus it should attempt to download everything
            self.assertEqual(station.stationxml_filename,
                             os.path.join("random", "TA.A001.xml"))
            self.assertEqual(station.stationxml_status,
                             STATUS.NEEDS_DOWNLOADING)
            self.assertEqual(station.have_station_information, {})
            self.assertEqual(station.want_station_information,
                             station.miss_station_information)
            self.assertEqual(station.want_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

            # Now it returns a filename, the filename exists, and it
            # contains all necessary information.
            ChannelAvailability = collections.namedtuple(
                "ChannelAvailability",
                ["network", "station", "location", "channel", "starttime",
                 "endtime", "filename"])
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            with mock.patch("os.path.exists") as exists_p:
                exists_p.return_value = True
                with mock.patch("obspy.clients.fdsn.download_helpers.utils."
                                "get_stationxml_contents") as c_patch:
                    c_patch.return_value = [
                        ChannelAvailability("TA", "A001", "", "BHZ",
                                            obspy.UTCDateTime(2013, 1, 1),
                                            obspy.UTCDateTime(2016, 1, 1), ""),
                        ChannelAvailability("TA", "A001", "00", "EHE",
                                            obspy.UTCDateTime(2013, 1, 1),
                                            obspy.UTCDateTime(2016, 1, 1), "")]
                    station.prepare_stationxml_download(
                        stationxml_storage="random", logger=logger)
            # It then should not attempt to download anything as everything
            # thats needed is already available.
            self.assertEqual(station.stationxml_status, STATUS.EXISTS)
            self.assertEqual(station.miss_station_information, {})
            self.assertEqual(station.want_station_information,
                             station.have_station_information)
            self.assertEqual(station.want_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

            # The last option for the simple case is that the file exists,
            # but it only contains part of the required information. In that
            # case everything will be downloaded again.
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            with mock.patch("os.path.exists") as exists_p:
                exists_p.return_value = True
                with mock.patch("obspy.clients.fdsn.download_helpers.utils."
                                "get_stationxml_contents") as c_patch:
                    c_patch.return_value = [
                        ChannelAvailability("TA", "A001", "", "BHZ",
                                            obspy.UTCDateTime(2013, 1, 1),
                                            obspy.UTCDateTime(2016, 1, 1), "")]
                    station.prepare_stationxml_download(
                        stationxml_storage="random", logger=logger)
            # It then should not attempt to download anything as everything
            # thats needed is already available.
            self.assertEqual(station.stationxml_status,
                             STATUS.NEEDS_DOWNLOADING)
            self.assertEqual(station.have_station_information, {})
            self.assertEqual(station.want_station_information,
                             station.miss_station_information)
            self.assertEqual(station.want_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

            # Now a combination that barely lacks the required time range.
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            with mock.patch("os.path.exists") as exists_p:
                exists_p.return_value = True
                with mock.patch("obspy.clients.fdsn.download_helpers.utils."
                                "get_stationxml_contents") as c_patch:
                    c_patch.return_value = [
                        ChannelAvailability(
                            "TA", "A001", "", "BHZ",
                            obspy.UTCDateTime(2015, 1, 1),
                            obspy.UTCDateTime(2015, 1, 1, 0, 9), ""),
                        ChannelAvailability("TA", "A001", "00", "EHE",
                                            obspy.UTCDateTime(2013, 1, 1),
                                            obspy.UTCDateTime(2016, 1, 1), "")]
                    station.prepare_stationxml_download(
                        stationxml_storage="random", logger=logger)
            # It then should not attempt to download anything as everything
            # thats needed is already available.
            self.assertEqual(STATUS.NEEDS_DOWNLOADING,
                             station.stationxml_status)
            self.assertEqual({}, station.have_station_information)
            self.assertEqual(station.want_station_information,
                             station.miss_station_information)
            self.assertEqual(station.want_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

    def test_prepare_stationxml_download_dictionary_case(self):
        """
        Tests the case of the prepare_stationxml_download() method if the
        get_stationxml_contents() method returns a dictionary.
        """
        logger = mock.MagicMock()
        temporal_bounds = (obspy.UTCDateTime(2015, 1, 1),
                           obspy.UTCDateTime(2015, 1, 1, 0, 10))

        def _create_station():
            st = obspy.UTCDateTime(2015, 1, 1)
            time_intervals = [
                TimeInterval(st + _i * 60, st + (_i + 1) * 60)
                for _i in range(10)]
            c1 = Channel(location="", channel="BHZ",
                         intervals=copy.deepcopy(time_intervals))
            c2 = Channel(location="00", channel="EHE",
                         intervals=copy.deepcopy(time_intervals))
            channels = [c1, c2]
            all_tis = []
            for chan in channels:
                all_tis.extend(chan.intervals)
            station = Station(network="TA", station="A001", latitude=1,
                              longitude=2, channels=channels)
            return station
        # Again mock the os.makedirs function as it is called quite a bit.
        with mock.patch("os.makedirs") as p:

            # Case 1: All channels are missing and thus must be downloaded.
            def stationxml_storage(network, station, channels, starttime,
                                   endtime):
                return {
                    "missing_channels": channels[:],
                    "available_channels": [],
                    "filename": os.path.join("random", "TA.AA01_.xml")
                }
                pass
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            station.prepare_stationxml_download(
                stationxml_storage=stationxml_storage, logger=logger)
            # The directory should have been created if it does not exists.
            self.assertEqual(p.call_count, 1)
            self.assertEqual(p.call_args[0][0], "random")
            self.assertEqual(station.stationxml_status,
                             STATUS.NEEDS_DOWNLOADING)
            self.assertEqual(station.have_station_information, {})
            self.assertEqual(station.want_station_information,
                             station.miss_station_information)
            self.assertEqual(station.want_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

            # Case 2: All channels are existing and thus nothing happens.
            def stationxml_storage(network, station, channels, starttime,
                                   endtime):
                return {
                    "missing_channels": [],
                    "available_channels": channels[:],
                    "filename": os.path.join("random", "TA.AA01_.xml")
                }
                pass
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            station.prepare_stationxml_download(
                stationxml_storage=stationxml_storage, logger=logger)
            # The directory should have been created if it does not exists.
            self.assertEqual(p.call_count, 1)
            self.assertEqual(p.call_args[0][0], "random")
            self.assertEqual(station.stationxml_status,
                             STATUS.EXISTS)
            self.assertEqual(station.miss_station_information, {})
            self.assertEqual(station.want_station_information,
                             station.have_station_information)
            self.assertEqual(station.have_station_information, {
                ("", "BHZ"): temporal_bounds,
                ("00", "EHE"): temporal_bounds
            })
            p.reset_mock()

            # Case 3: Mixed case.
            def stationxml_storage(network, station, channels, starttime,
                                   endtime):
                return {
                    "missing_channels":
                        [_i for _i in channels if _i[0] == "00"],
                    "available_channels":
                        [_i for _i in channels if _i[0] == ""],
                    "filename": os.path.join("random", "TA.AA01_.xml")
                }
                pass
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            station.prepare_stationxml_download(
                stationxml_storage=stationxml_storage, logger=logger)
            # The directory should have been created if it does not exists.
            self.assertEqual(p.call_count, 1)
            self.assertEqual(p.call_args[0][0], "random")
            self.assertEqual(station.stationxml_status,
                             STATUS.NEEDS_DOWNLOADING)
            self.assertEqual(station.miss_station_information, {
                ("00", "EHE"): temporal_bounds
            })
            self.assertEqual(station.have_station_information, {
                ("", "BHZ"): temporal_bounds
            })
            self.assertEqual(station.want_station_information, {
                ("00", "EHE"): temporal_bounds,
                ("", "BHZ"): temporal_bounds
            })
            p.reset_mock()

            # Case 4: The stationxml_storage() function does not return all
            # required information. A warning should thus be raised.
            logger.reset_mock()

            def stationxml_storage(network, station, channels, starttime,
                                   endtime):
                return {
                    "missing_channels":
                        [_i for _i in channels if _i[0] == "00"],
                    "available_channels": [],
                    "filename": os.path.join("random", "TA.AA01_.xml")
                }
                pass
            station = _create_station()
            for cha in station.channels:
                for ti in cha.intervals:
                    ti.status = STATUS.DOWNLOADED
            station.prepare_stationxml_download(
                stationxml_storage=stationxml_storage, logger=logger)
            # The directory should have been created if it does not exists.
            self.assertEqual(p.call_count, 1)
            self.assertEqual(p.call_args[0][0], "random")
            self.assertEqual(station.stationxml_status,
                             STATUS.NEEDS_DOWNLOADING)
            self.assertEqual(station.miss_station_information, {
                ("00", "EHE"): temporal_bounds
            })
            self.assertEqual(station.have_station_information, {})
            self.assertEqual(station.want_station_information, {
                ("00", "EHE"): temporal_bounds,
                ("", "BHZ"): temporal_bounds
            })
            self.assertEqual(logger.method_calls[0][0], "warning")
            self.assertTrue(
                "did not return information about channels" in
                logger.method_calls[0][1][0])
            self.assertTrue(
                "BHZ" in
                logger.method_calls[0][1][0])

    def test_str_method(self):
        """
        Test the __str__ method of the Station object.
        """
        # Minimal information.
        st = Station("BW", "ALTM", 10, 20, [])
        self.assertEqual(str(st), (
            "Station 'BW.ALTM' [Lat: 10.00, Lng: 20.00]\n"
            "\t-> Filename: None (does not yet exist)\n"
            "\t-> Wants station information for channels:  \n"
            "\t-> Has station information for channels:    \n"
            "\t-> Misses station information for channels: \n\t"))

        # A bit more information.
        channels = [Channel("", "BHE", [TimeInterval(obspy.UTCDateTime(0),
                                                     obspy.UTCDateTime(10))]),
                    Channel("", "BHZ", [TimeInterval(obspy.UTCDateTime(10),
                                                     obspy.UTCDateTime(20))])]
        st = Station("BW", "ALTM", 10, 20, channels=channels,
                     stationxml_status=STATUS.ignore)
        self.assertEqual(str(st), (
            "Station 'BW.ALTM' [Lat: 10.00, Lng: 20.00]\n"
            "\t-> Filename: None (does not yet exist)\n"
            "\t-> Wants station information for channels:  \n"
            "\t-> Has station information for channels:    \n"
            "\t-> Misses station information for channels: \n"
            "\tChannel '.BHE':\n"
            "\t\tTimeInterval(start=UTCDateTime(1970, 1, 1, 0, 0), "
            "end=UTCDateTime(1970, 1, 1, 0, 0, 10), filename=None, "
            "status='none')\n"
            "\tChannel '.BHZ':\n"
            "\t\tTimeInterval(start=UTCDateTime(1970, 1, 1, 0, 0, 10), "
            "end=UTCDateTime(1970, 1, 1, 0, 0, 20), filename=None, "
            "status='none')"))


class ClientDownloadHelperTestCase(unittest.TestCase):
    """
    Test cases for the ClientDownloadHelper class.
    """
    def setUp(self):
        self.client = mock.MagicMock()
        self.client_name = "Test"
        self.restrictions = None
        self.domain = None
        self.mseed_storage = "miniseed"
        self.stationxml_storage = "stationxml_storage"
        self.logger = mock.MagicMock()

    def _init_client(self):
        return ClientDownloadHelper(
            client=self.client, client_name=self.client_name,
            restrictions=self.restrictions, domain=self.domain,
            mseed_storage=self.mseed_storage,
            stationxml_storage=self.stationxml_storage, logger=self.logger)

    def test_looped_methods(self):
        """
        Some methods are just used to loop over methods on the station
        objects. Those are mocked here.
        """
        sta1 = mock.MagicMock()
        sta2 = mock.MagicMock()

        c = self._init_client()
        c.stations["BW.ALTM"] = sta1
        c.stations["BW.RJOB"] = sta2

        c.prepare_mseed_download()
        self.assertEqual(sta1.prepare_mseed_download.call_count, 1)
        self.assertEqual(sta2.prepare_mseed_download.call_count, 1)
        self.assertEqual(
            sta1.prepare_mseed_download.call_args[1]["mseed_storage"],
            self.mseed_storage)
        self.assertEqual(
            sta2.prepare_mseed_download.call_args[1]["mseed_storage"],
            self.mseed_storage)

        sta1.reset_mock()
        sta2.reset_mock()

        c.prepare_stationxml_download()
        self.assertEqual(sta1.prepare_stationxml_download.call_count, 1)
        self.assertEqual(sta2.prepare_stationxml_download.call_count, 1)

        sta1.reset_mock()
        sta2.reset_mock()

        c.sanitize_downloads()
        self.assertEqual(sta1.sanitize_downloads.call_count, 1)
        self.assertEqual(sta2.sanitize_downloads.call_count, 1)

    def test_station_list_nearest_neighbour_filter(self):
        """
        Test the filtering based on geographical distance.
        """
        self.restrictions = Restrictions(
            0, 1, minimum_interstation_distance_in_m=250)

        def _m_to_deg(meters):
            return meters / 111000.0

        c = self._init_client()

        # Only the one at longitude 200 should be removed as it is the only one
        # that has two neighbours inside the filter radius.
        c.stations = {
            ("A", "A"): Station("A", "A", 0, _m_to_deg(0), []),
            ("B", "B"): Station("B", "B", 0, _m_to_deg(200), []),
            ("C", "C"): Station("C", "C", 0, _m_to_deg(400), []),
            ("D", "D"): Station("D", "D", 0, _m_to_deg(2000), [])
        }
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual([("A", "A"), ("C", "C"), ("D", "D")],
                         sorted(c.stations.keys()))

        # The two at 200 and 250 m longitude should be removed.
        c.stations = {
            ("A", "A"): Station("A", "A", 0, _m_to_deg(0), []),
            ("B", "B"): Station("B", "B", 0, _m_to_deg(200), []),
            ("C", "C"): Station("C", "C", 0, _m_to_deg(250), []),
            ("D", "D"): Station("D", "D", 0, _m_to_deg(400), []),
            ("E", "E"): Station("E", "E", 0, _m_to_deg(2000), [])}
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual([("A", "A"), ("D", "D"), ("E", "E")],
                         sorted(c.stations.keys()))

        # Set the distance to 1 degree and check the longitude behaviour at
        # the longitude wraparound point.
        stations = {
            ("A", "A"): Station("A", "A", 0, 0, []),
            ("B", "B"): Station("B", "B", 0, 90, []),
            ("C", "C"): Station("C", "C", 0, 180, []),
            ("D", "D"): Station("D", "D", 0, -90, []),
            ("E", "E"): Station("E", "E", 0, -180, [])}

        self.restrictions = Restrictions(
            0, 1, minimum_interstation_distance_in_m=111000)
        c = self._init_client()
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])

        # Only 4 stations should remain and either the one at 0,180 or the
        # one at 0, -180 should have been removed as they are equal.
        self.assertEqual(len(c.stations), 4)
        self.assertTrue(
            sorted(c.stations.keys()) == [("A", "A"), ("B", "B"), ("C", "C"),
                                          ("D", "D")] or
            sorted(c.stations.keys()) == [("A", "A"), ("B", "B"), ("D", "D"),
                                          ("E", "E")])

        # Test filtering around the longitude wraparound.
        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, 179.2, []),
            ("C", "C"): Station("C", "C", 0, 180.8, [])}
        # The middle one should be removed as then the other two can be kept.
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual([("B", "B"), ("C", "C")],
                         sorted(c.stations.keys()))
        # Same but longitude defined the other way around.
        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, 179.2, []),
            ("C", "C"): Station("C", "C", 0, -179.2, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual([("B", "B"), ("C", "C")],
                         sorted(c.stations.keys()))

        # Test the conversion of lat/lng to meter distances.
        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, -180, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, -179.5, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, -179.1, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, 180, []),
            ("B", "B"): Station("B", "B", 0, 178.9, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 2)

        # Also test the latitude settings.
        stations = {
            ("A", "A"): Station("A", "A", 0, -90, []),
            ("B", "B"): Station("B", "B", 0, -90, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, -90, []),
            ("B", "B"): Station("B", "B", 0, -89.5, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, -90, []),
            ("B", "B"): Station("B", "B", 0, -89.1, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 1)

        stations = {
            ("A", "A"): Station("A", "A", 0, -90, []),
            ("B", "B"): Station("B", "B", 0, -88.9, [])}
        c.stations = stations
        c.filter_stations_based_on_minimum_distance([])
        self.assertEqual(len(c.stations), 2)


class DownloadHelperTestCase(unittest.TestCase):
    """
    Test cases for the DownloadHelper class.
    """
    @mock.patch("obspy.clients.fdsn.download_helpers.download_helpers."
                "DownloadHelper._initialize_clients")
    def test_initialization(self, patch):
        """
        Tests the initialization of the DownloadHelper object.
        """
        d = DownloadHelper()
        self.assertEqual(patch.call_count, 1)
        # The amount of services is variable and more and more get added.
        # Assert its larger then 8 and contains a couple stable ones.
        self.assertTrue(len(d.providers) > 8)
        self.assertTrue("IRIS" in d.providers)
        self.assertTrue("ORFEUS" in d.providers)
        patch.reset_mock()

        d = DownloadHelper(providers=["A", "B", "IRIS"])
        self.assertEqual(patch.call_count, 1)
        self.assertEqual(d.providers, ("A", "B", "IRIS"))
        patch.reset_mock()


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.makeSuite(DomainTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(DownloadHelpersUtilTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(TimeIntervalTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(ChannelTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(StationTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(DownloadHelperTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(ClientDownloadHelperTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(RestrictionsTestCase, 'test'))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
