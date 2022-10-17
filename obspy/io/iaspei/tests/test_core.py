# -*- coding: utf-8 -*-
import io
import os
import re
import unittest

from obspy import read_events
from obspy.io.iaspei.core import _read_ims10_bulletin, _is_ims10_bulletin


data_dir = os.path.join(os.path.dirname(__file__), 'data')
path_to_quakeml = os.path.join(data_dir, '19670130012028.xml')
catalog_from_quakeml = os.path.join(data_dir, '19670130012028.xml')


def _assert_catalog(got):
    got_id_prefix = str(got.resource_id).encode('UTF-8')
    # first of all, replace the random hash in our test file with the prefix
    # that was used during reading the ISF file
    with open(path_to_quakeml, 'rb') as fh:
        data = fh.read()
    match = re.search(b'publicID="(smi:local/[a-z0-9-]*)"', data)
    expected_id_prefix = match.group(1)
    data, num_subs = re.subn(expected_id_prefix, got_id_prefix, data)
    # 49 resource id replacements should be done in the QuakeML file we compare
    # against
    assert num_subs == 49
    bio = io.BytesIO(data)
    expected = read_events(bio, format="QUAKEML")
    # now first check if we got the expected number of picks and station
    # magnitudes, because the quakeml file has been stripped of picks and
    # station magnitudes to save space and time
    assert len(got[0].picks) == 255
    assert len(got[0].station_magnitudes) == 15
    # ok now crop the got catalog accordingly, afterwards it should compare
    # equal to our comparison catalog
    got[0].picks = got[0].picks[:4]
    got[0].station_magnitudes = got[0].station_magnitudes[:7]
    # # now we also have to replace comment ids in both catalogs..
    # for cat in (got, expected):
    #     for item in cat.events + cat[0].origins + cat[0].magnitudes + \
    #             cat[0].station_magnitudes + cat[0].picks:
    #         for comment in item.comments:
    #             comment.resource_id = 'smi:local/dummy'

    # some more fixes for the comparison, these are due to buggy QuakeML reader
    # behavior and should be fixed in io.quakeml eventually
    for event in expected:
        for pick in event.picks:
            # QuakeML reader seems to set `network_code=""` if it's not in the
            # xml file.. account for this strange behavior for this test
            pick.waveform_id.network_code = None
            # QuakeML reader seems to add empty QuantityError for
            # pick.horizontal_slowness_errors
            for key in ['horizontal_slowness_errors', 'time_errors',
                        'backazimuth_errors']:
                setattr(pick, key, None)
        for origin in event.origins:
            if origin.origin_uncertainty is not None:
                # QuakeML reader seems to add empty ConfidenceEllipsoid
                origin.origin_uncertainty.confidence_ellipsoid = None
            # QuakeML reader seems to add empty QuantityError for
            # pick.horizontal_slowness_errors
            for key in ['time_errors', 'longitude_errors', 'latitude_errors',
                        'depth_errors']:
                setattr(origin, key, None)
        for station_magnitude in event.station_magnitudes:
            # QuakeML reader seems to set origin_id to
            # `ResourceIdentifier(id="None")`
            # QuakeML reader seems to add empty QuantityError for
            # pick.horizontal_slowness_errors
            for key in ['origin_id', 'mag_errors']:
                setattr(station_magnitude, key, None)
        for magnitude in event.magnitudes:
            # QuakeML reader seems to add empty QuantityError for
            # pick.horizontal_slowness_errors
            for key in ['mag_errors']:
                setattr(magnitude, key, None)
    # now finally these catalogs should compare equal
    assert got == expected


class IASPEITestCase(unittest.TestCase):
    """
    Test suite for obspy.io.iaspei.core
    """
    def setUp(self):
        self.data_dir = data_dir
        # XXX the converted QuakeML file is not complete.. many picks and
        # station magnitudes were removed from it to save space.
        self.path_to_ims = os.path.join(self.data_dir, '19670130012028.isf')

    def test_reading(self):
        """
        Test reading IMS10 bulletin format
        """
        cat = _read_ims10_bulletin(self.path_to_ims, _no_uuid_hashes=True)
        self.assertEqual(len(cat), 1)
        _assert_catalog(cat)

    def test_reading_via_file(self):
        """
        Test reading IMS10 bulletin format from open files.
        """
        with io.open(self.path_to_ims, "rb") as fh:
            cat = _read_ims10_bulletin(fh, _no_uuid_hashes=True)
        self.assertEqual(len(cat), 1)
        _assert_catalog(cat)

        with io.open(self.path_to_ims, "rt", encoding="UTF-8") as fh:
            cat = _read_ims10_bulletin(fh, _no_uuid_hashes=True)
        self.assertEqual(len(cat), 1)
        _assert_catalog(cat)

    def test_reading_via_bytes_io(self):
        """
        Test reading IMS10 bulletin format from bytes io object.
        """
        with io.open(self.path_to_ims, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                cat = _read_ims10_bulletin(buf, _no_uuid_hashes=True)
        self.assertEqual(len(cat), 1)
        _assert_catalog(cat)

    def test_reading_via_plugin(self):
        """
        Test reading IMS10 bulletin format
        """
        cat = read_events(self.path_to_ims, format='IMS10BULLETIN',
                          _no_uuid_hashes=True)
        self.assertEqual(len(cat), 1)
        _assert_catalog(cat)

    def test_is_ims10_bulltin(self):
        """
        Test checking if file is IMS10 bulletin format
        """
        self.assertTrue(_is_ims10_bulletin(self.path_to_ims))
        self.assertFalse(_is_ims10_bulletin(path_to_quakeml))

    def test_is_ims10_bulltin_open_file(self):
        with open(self.path_to_ims, "rb") as fh:
            self.assertTrue(_is_ims10_bulletin(fh))

        with open(self.path_to_ims, "rt", encoding="utf-8") as fh:
            self.assertTrue(_is_ims10_bulletin(fh))

        with open(path_to_quakeml, "rb") as fh:
            self.assertFalse(_is_ims10_bulletin(fh))

        with open(path_to_quakeml, "rt", encoding="utf-8") as fh:
            self.assertFalse(_is_ims10_bulletin(fh))

    def test_is_ims10_bulltin_from_bytes_io(self):
        with open(self.path_to_ims, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                self.assertTrue(_is_ims10_bulletin(buf))

        with open(path_to_quakeml, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                self.assertFalse(_is_ims10_bulletin(buf))
