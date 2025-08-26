# -*- coding: utf-8 -*-
import io
import re

import pytest
import warnings
from obspy import read_events
from obspy.io.iaspei.core import _read_ims10_bulletin, _is_ims10_bulletin
from obspy.core.event import ResourceIdentifier


class TestIASPEI():
    """
    Test suite for obspy.io.iaspei.core
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        # XXX the converted QuakeML file is not complete.. many picks and
        # station magnitudes were removed from it to save space.
        self.path_to_ims = testdata['19670130012028.isf']
        self.path_to_quakeml = testdata['19670130012028.xml']
        self.path_to_ims_2 = testdata["ipe202409sel_ims.txt"]
        self.path_to_quakeml_2 = testdata["ipe202409sel_ims.xml"]

    def prepare_comparison(self, catalog):
        # some more fixes for the comparison, these are due to buggy QuakeML
        # reader behavior and should be fixed in io.quakeml eventually
        for event in catalog:
            for pick in event.picks:
                # QuakeML reader seems to set `network_code=""` if it's not in
                # the xml file.. account for this strange behavior for this
                # test
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
                for key in ['time_errors', 'longitude_errors',
                            'latitude_errors', 'depth_errors']:
                    setattr(origin, key, None)
            for station_magnitude in event.station_magnitudes:
                setattr(station_magnitude.waveform_id, 'network_code', None)
                # QuakeML reader seems to set origin_id to
                # `ResourceIdentifier(id="None")`
                if station_magnitude.origin_id == ResourceIdentifier(
                        id="None"):
                    setattr(station_magnitude, 'origin_id', None)
                # QuakeML reader seems to add empty QuantityError for
                # pick.horizontal_slowness_errors
                for key in ['mag_errors']:
                    setattr(station_magnitude, key, None)
            for magnitude in event.magnitudes:
                # QuakeML reader seems to add empty QuantityError for
                # pick.horizontal_slowness_errors
                for key in ['mag_errors']:
                    setattr(magnitude, key, None)

    def _assert_catalog(self, got):
        got_id_prefix = str(got.resource_id).encode('UTF-8')
        # first of all, replace the random hash in our test file with the
        # prefix that was used during reading the ISF file
        with open(self.path_to_quakeml, 'rb') as fh:
            data = fh.read()
        match = re.search(b'publicID="(smi:local/[a-z0-9-]*)"', data)
        expected_id_prefix = match.group(1)
        data, num_subs = re.subn(expected_id_prefix, got_id_prefix, data)
        # resource id replacements should be done in the QuakeML file we
        # compare against
        assert num_subs == 65
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
        for origin in got[0].origins:
            if origin.arrivals:
                origin.arrivals = origin.arrivals[:4]
        self.prepare_comparison(expected)
        # now finally these catalogs should compare equal
        assert got == expected

    def _assert_catalog2(self, got):
        # Unify the random hash prefix of all resource_ids on the side
        # of the ISF test file and on the QuakeML model file side.
        # Resource id replacements should be done in the QuakeML file.
        got_id_prefix = str(got.resource_id).encode('UTF-8')
        with open(self.path_to_quakeml_2, 'rb') as fh:
            data = fh.read()
        match = re.search(b'publicID="(smi:local/[a-z0-9-]*)"', data)
        expected_id_prefix = match.group(1)
        data, num_subs = re.subn(expected_id_prefix, got_id_prefix, data)
        bio = io.BytesIO(data)
        expected = read_events(bio, format="QUAKEML")
        self.prepare_comparison(expected)
        # take the entire uncropped got catalog and compare against expected
        assert got == expected

    def test_reading(self):
        """
        Test reading IMS10 bulletin format
        """
        cat = _read_ims10_bulletin(self.path_to_ims, _no_uuid_hashes=True)
        assert len(cat) == 1
        self._assert_catalog(cat)

    def test_reading_via_file(self):
        """
        Test reading IMS10 bulletin format from open files.
        """
        with io.open(self.path_to_ims, "rb") as fh:
            cat = _read_ims10_bulletin(fh, _no_uuid_hashes=True)
        assert len(cat) == 1
        self._assert_catalog(cat)

        with io.open(self.path_to_ims, "rt", encoding="UTF-8") as fh:
            cat = _read_ims10_bulletin(fh, _no_uuid_hashes=True)
        assert len(cat) == 1
        self._assert_catalog(cat)

    def test_reading_via_bytes_io(self):
        """
        Test reading IMS10 bulletin format from bytes io object.
        """
        with io.open(self.path_to_ims, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                cat = _read_ims10_bulletin(buf, _no_uuid_hashes=True)
        assert len(cat) == 1
        self._assert_catalog(cat)

    def test_reading_via_plugin(self):
        """
        Test reading IMS10 bulletin format
        """
        cat = read_events(self.path_to_ims, format='IMS10BULLETIN',
                          _no_uuid_hashes=True)
        assert len(cat) == 1
        self._assert_catalog(cat)

    def test_is_ims10_bulltin(self):
        """
        Test checking if file is IMS10 bulletin format
        """
        assert _is_ims10_bulletin(self.path_to_ims)
        assert not _is_ims10_bulletin(self.path_to_quakeml)

    def test_is_ims10_bulltin_open_file(self):
        with open(self.path_to_ims, "rb") as fh:
            assert _is_ims10_bulletin(fh)

        with open(self.path_to_ims, "rt", encoding="utf-8") as fh:
            assert _is_ims10_bulletin(fh)

        with open(self.path_to_quakeml, "rb") as fh:
            assert not _is_ims10_bulletin(fh)

        with open(self.path_to_quakeml, "rt", encoding="utf-8") as fh:
            assert not _is_ims10_bulletin(fh)

    def test_is_ims10_bulltin_from_bytes_io(self):
        with open(self.path_to_ims, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                assert _is_ims10_bulletin(buf)

        with open(self.path_to_quakeml, "rb") as fh:
            with io.BytesIO(fh.read()) as buf:
                buf.seek(0, 0)
                assert not _is_ims10_bulletin(buf)

    def test_reading_2(self):
        """
        Test reading IMS10 bulletin format
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # The last of the three events
            # has a phase block with incorrect #OriginID tag
            cat = _read_ims10_bulletin(self.path_to_ims_2, _no_uuid_hashes=True)
            assert len(w) == 3
            assert issubclass(w[-3].category, UserWarning)
            assert "Phase block cannot be fully processed" in str(w[-3].message)
            assert issubclass(w[-2].category, UserWarning)
            assert "This pick would have a time more than 6 hours after" in str(w[-2].message)
            assert issubclass(w[-1].category, UserWarning)
            assert "Could not determine absolute time of pick" in str(w[-1].message)
        assert len(cat) == 3
        self._assert_catalog2(cat)
