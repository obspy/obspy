# -*- coding: utf-8 -*-
import os
import unittest

from obspy.core.event import read_events
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, get_example_file
from obspy.io.zmap import core as zmap


_STD_ZMAP_FIELDS = ('lon', 'lat', 'year', 'month', 'day', 'mag', 'depth',
                    'hour', 'minute', 'second')
_EXT_ZMAP_FIELDS = ('h_err', 'z_err', 'm_err')
_ORIGIN_FIELDS = ('lon', 'lat', 'year', 'month', 'day', 'depth', 'hour',
                  'minute', 'second', 'h_err', 'z_err')
_MAGNITUDE_FIELDS = ('mag', 'm_err')


class ZMAPTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.zmap.core
    """
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        path_to_catalog = os.path.join(data_dir, 'neries_events.xml')
        self.catalog = read_events(path_to_catalog)
        self.zmap_fields = _STD_ZMAP_FIELDS
        # Extract our favorite test event from the catalog
        test_event_id = 'quakeml:eu.emsc/event/20120404_0000041'
        self.test_event = next(e for e in self.catalog.events
                               if e.resource_id.id == test_event_id)
        self.test_data = {
            'lon': '79.689000', 'lat': '41.818000', 'month': '4',
            'year': '2012.258465590847', 'day': '4', 'hour': '14',
            'minute': '21', 'second': '42.3', 'depth': '1.000000',
            'mag': '4.400000'
        }

    def tearDown(self):
        # Make sure events are deleted before the next test to prevent
        # resource identifier warnings
        self.catalog = None
        self.test_event = None

    def test_serialize(self):
        """
        Test serialization to zmap format
        """
        pickler = zmap.Pickler()
        # test full event (including origin/magnitude)
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)
        self.assertEqual(dump.count('\n'), 3)
        # no preferred origin -- still dump first origin
        oid = self.test_event.preferred_origin_id
        self.test_event.preferred_origin_id = None
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)
        self.test_event.preferred_origin_id = oid
        # no preferred magnitude -- still dump first magnitude
        mid = self.test_event.preferred_origin_id
        self.test_event.preferred_magnitude_id = None
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)
        self.test_event.preferred_magnitude_id = mid

    def test_plugin_interface(self):
        """
        Test if zmap writing works via obspy's plugin interface
        """
        with NamedTemporaryFile() as f:
            self.catalog.write(f, format='ZMAP')
            f.seek(0)
            file_content = f.read().decode('utf-8')
        self.assertIn(self._expected_string(self.test_data), file_content)

    def test_dump_to_file(self):
        """
        Test output to pre-opened file
        """
        with NamedTemporaryFile() as f:
            zmap._write_zmap(self.catalog, f)
            f.seek(0)
            file_content = f.read().decode('utf-8')
        self.assertIn(self._expected_string(self.test_data), file_content)

    def test_dump_to_filename(self):
        """
        Test output to file with a filename specified
        """
        with NamedTemporaryFile() as f:
            zmap._write_zmap(self.catalog, f.name)
            f.seek(0)
            file_content = f.read().decode('utf-8')
        self.assertIn(self._expected_string(self.test_data), file_content)

    def test_dump_with_uncertainty(self):
        """
        Test export of non-standard (CSEP) uncertainty fields
        """
        self.zmap_fields += _EXT_ZMAP_FIELDS
        self.test_data.update({'h_err': 'NaN', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)

    def test_ou_hz_error(self):
        """
        Test hz error extraction from origin_uncertainty
        """
        self.zmap_fields += _EXT_ZMAP_FIELDS
        self.test_data.update({'h_err': '1.000000', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        o = self.test_event.preferred_origin()
        o.origin_uncertainty.preferred_description = 'horizontal uncertainty'
        o.origin_uncertainty.horizontal_uncertainty = 1.0
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)
        # with unsupported preferred_description
        self.test_data.update({'h_err': 'NaN', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        o.origin_uncertainty.preferred_description = 'uncertainty ellipse'
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)

    def test_lat_lon_hz_error(self):
        """
        Test hz error extraction from lat/lon
        """
        self.zmap_fields += _EXT_ZMAP_FIELDS
        self.test_data.update({'h_err': '0.138679', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        o = self.test_event.preferred_origin()
        o.latitude_errors.uncertainty = .001
        o.longitude_errors.uncertainty = .001
        dump = pickler.dumps(self.catalog)
        self.assertIn(self._expected_string(self.test_data), dump)

    def test_is_zmap(self):
        """
        Test zmap format detection
        """
        # Regular ZMAP
        test_events = [self.test_data, dict(self.test_data, mag='5.1')]
        with NamedTemporaryFile() as f:
            f.write(self._serialize(test_events).encode('utf-8'))
            self.assertTrue(zmap._is_zmap(f.name))
            # Pre-opened file
            f.seek(0)
            self.assertTrue(zmap._is_zmap(f))
        # Extended ZMAP (13 columns)
        self.zmap_fields += _EXT_ZMAP_FIELDS
        self.test_data.update({'h_err': '0.138679', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        test_events = [self.test_data, dict(self.test_data, mag='5.1')]
        with NamedTemporaryFile() as f:
            f.write(self._serialize(test_events).encode('utf-8'))
            self.assertTrue(zmap._is_zmap(f.name))
        # ZMAP string
        test_string = self._serialize(test_events)
        self.assertTrue(zmap._is_zmap(test_string))
        # Non-ZMAP string
        test_string = '0.000000\t' + test_string
        self.assertFalse(zmap._is_zmap(test_string + '\n'))
        # Non-ZMAP file (14 columns)
        self.zmap_fields += ('dummy',)
        self.test_data.update({'dummy': '0'})
        test_events = [self.test_data, dict(self.test_data, mag='5.1')]
        with NamedTemporaryFile() as f:
            f.write(self._serialize(test_events).encode('utf-8'))
            self.assertFalse(zmap._is_zmap(f.name))
        # Non-ZMAP file (non-numeric columns)
        self.zmap_fields = _STD_ZMAP_FIELDS + _EXT_ZMAP_FIELDS
        self.test_data.update({'mag': 'bad'})
        test_events = [self.test_data]
        with NamedTemporaryFile() as f:
            f.write(self._serialize(test_events).encode('utf-8'))
            self.assertFalse(zmap._is_zmap(f.name))

    def test_is_zmap_binary_files(self):
        """
        Test zmap format detection on non-ZMAP (e.g. binary) files, see #1022.
        """
        # Non-ZMAP file, binary
        for filename in ["test.mseed", "test.sac"]:
            file_ = get_example_file(filename)
            self.assertFalse(zmap._is_zmap(file_))

    def test_deserialize(self):
        """
        Test ZMAP deserialization to catalog
        """
        # Regular ZMAP
        test_events = [self.test_data, dict(self.test_data, mag='5.1')]
        zmap_str = self._serialize(test_events)
        catalog = zmap.Unpickler().loads(zmap_str)
        self._assert_zmap_equal(catalog, test_events)
        # Leniency (1 to 13 or more columns (extra columns are ignored))
        self.zmap_fields += _EXT_ZMAP_FIELDS + ('extra',)
        self.test_data.update({'h_err': '0.138679', 'z_err': '0.000000',
                               'm_err': '0.000000', 'extra': '0.000000'})
        data = {}
        for field in self.zmap_fields:
            data[field] = self.test_data[field]
            test_events = [data, dict(data, lon='0')]
            zmap_str = self._serialize(test_events, fill_nans=False)
            catalog = zmap.Unpickler().loads(zmap_str)
            self._assert_zmap_equal(catalog, test_events)
        # Deserialize accepts a year without the weird fractional part that
        # redundantly defines the point in time within the year.
        test_events = [dict(e, year=int(float(e['year'])))
                       for e in test_events]
        zmap_str = self._serialize((test_events))
        catalog = zmap.Unpickler().loads(zmap_str)
        self._assert_zmap_equal(catalog, test_events)

    def test_read(self):
        # via file, file name, plugin interface
        test_events = [self.test_data, dict(self.test_data, lon='5.1')]
        zmap_str = self._serialize((test_events))
        with NamedTemporaryFile() as f:
            f.write(zmap_str.encode('utf-8'))
            catalog = zmap._read_zmap(f.name)
            self._assert_zmap_equal(catalog, test_events)
            f.seek(0)
            catalog = zmap._read_zmap(f)
            self._assert_zmap_equal(catalog, test_events)
            catalog = read_events(f.name)
            self._assert_zmap_equal(catalog, test_events)
        # direct ZMAP string
        catalog = zmap._read_zmap(zmap_str)
        self._assert_zmap_equal(catalog, test_events)

    def test_read_float_seconds(self):
        """
        Test that floating point part of seconds is parsed correctly.
        """
        catalog = zmap._read_zmap(os.path.join(self.data_dir, "templates.txt"))
        self.assertEqual(catalog[0].origins[0].time.microsecond, 840000)
        self.assertEqual(catalog[1].origins[0].time.microsecond, 880000)
        self.assertEqual(catalog[2].origins[0].time.microsecond, 550000)
        self.assertEqual(catalog[3].origins[0].time.microsecond, 450000)

    def _assert_zmap_equal(self, catalog, dicts):
        """
        Compares a zmap imported catalog with test event dictionaries
        """
        self.assertEqual(len(catalog), len(dicts))
        for event, test_dict in zip(catalog, dicts):
            origin = event.preferred_origin()
            if any(k in test_dict for k in _ORIGIN_FIELDS):
                self.assertNotEqual(None, origin)
            magnitude = event.preferred_magnitude()
            if any(k in test_dict for k in _MAGNITUDE_FIELDS):
                self.assertNotEqual(None, magnitude)
            d = dict((k, float(v) if v != 'NaN' else None)
                     for (k, v) in test_dict.items())
            if 'lon' in d:
                self.assertEqual(d['lon'], origin.longitude)
            if 'lat' in d:
                self.assertEqual(d['lat'], origin.latitude)
            if 'depth' in d:
                self.assertEqual(d['depth'] * 1000, origin.depth)
            if 'z_err' in d:
                self.assertEqual(d['z_err'] * 1000, origin.depth_errors.
                                 uncertainty)
            if 'h_err' in d:
                self.assertEqual(d['h_err'], origin.origin_uncertainty
                                 .horizontal_uncertainty)
                self.assertEqual('horizontal uncertainty', origin
                                 .origin_uncertainty.preferred_description)
            if 'year' in d:
                year = d['year']
                comps = ['year', 'month', 'day', 'hour', 'minute', 'second']
                if year % 1 != 0:
                    start = UTCDateTime(int(year), 1, 1)
                    end = UTCDateTime(int(year) + 1, 1, 1)
                    utc = start + (year % 1) * (end - start)
                elif any(d.get(k, 0) > 0 for k in comps[1:]):
                    utc = UTCDateTime(*[
                        k == 'second' and d.get(k) or int(d.get(k))
                        for k in comps])
                self.assertEqual(utc, event.preferred_origin().time)
            if 'mag' in d:
                self.assertEqual(d['mag'], magnitude.mag)
            if 'm_err' in d:
                self.assertEqual(d['m_err'], magnitude.mag_errors.uncertainty)

    def _serialize(self, test_dicts, fill_nans=True):
        zmap_str = ''
        for d in test_dicts:
            if fill_nans:
                zmap_str += '\t'.join(str(d[f]) if f in d else 'NaN'
                                      for f in self.zmap_fields) + '\n'
            else:
                zmap_str += '\t'.join(str(d[f]) for f in self.zmap_fields
                                      if f in d) + '\n'
        return zmap_str

    def _expected_string(self, zmap_dict):
        """
        Returns the expected string from a ZMAP dump.

        zmap_dict contains (string) values for all the fields that are expected
        to have specific values. All other fields default to 'NaN'.
        """
        full_zmap = dict.fromkeys(self.zmap_fields, 'NaN')
        full_zmap.update(zmap_dict)
        string = '\t'.join(full_zmap[f] for f in self.zmap_fields)
        return string
