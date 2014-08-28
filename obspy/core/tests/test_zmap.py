# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core import zmap
from obspy.core.event import readEvents
from obspy.core.util import NamedTemporaryFile

import unittest
import os


class ZMAPTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.zmap
    """
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        path_to_catalog = os.path.join(data_dir, 'neries_events.xml')
        self.catalog = readEvents(path_to_catalog)
        self.zmap_fields = ['lon', 'lat', 'year', 'month', 'day', 'mag',
                            'depth', 'hour', 'minute', 'second']
        # Extract our favorite test event from the catalog
        test_event_id = 'quakeml:eu.emsc/event/20120404_0000041'
        self.test_event = next(e for e in self.catalog.events
                               if e.resource_id.id == test_event_id)
        self.test_data = {
            'lon': '79.689000', 'lat': '41.818000', 'month': '4',
            'year': '2012.258465590847', 'day': '4', 'hour': '14',
            'minute': '21', 'second': '42', 'depth': '1.000000',
            'mag': '4.400000'
        }

    def tearDown(self):
        # Not sure exactly why this is required. Without this event.py
        # starts to spew warnings about already existing resource identifiers
        # after the first test.
        self.catalog = None
        self.test_event = None

    def test_serialize(self):
        """
        Test serialization to zmap format
        """
        pickler = zmap.Pickler()
        # test full event (including origin/magnitude)
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string(self.test_data) in dump)
        self.assertEqual(dump.count('\n'), 3)
        # no preferred origin
        oid = self.test_event.preferred_origin_id
        self.test_event.preferred_origin_id = None
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string({'mag': '4.400000'}) in dump)
        self.test_event.preferred_origin_id = oid
        # no preferred magnitude
        self.test_event.preferred_magnitude_id = None
        dump = pickler.dumps(self.catalog)
        test_data = self.test_data.copy()
        del test_data['mag']
        self.assertTrue(self._expected_string(test_data) in dump)

    def test_dump_to_file(self):
        """
        Test output to pre-opened file
        """
        f = NamedTemporaryFile()
        zmap.writeZmap(self.catalog, f)
        f.seek(0)
        file_content = f.read().decode('utf-8')
        self.assertTrue(self._expected_string(self.test_data) in file_content)

    def test_dump_to_filename(self):
        """
        Test output to file with a filename specified
        """
        f = NamedTemporaryFile()
        zmap.writeZmap(self.catalog, f.name)
        f.seek(0)
        file_content = f.read().decode('utf-8')
        self.assertTrue(self._expected_string(self.test_data) in file_content)

    def test_dump_with_uncertainty(self):
        """
        Test export of non-standard (CSEP) uncertainty fields
        """
        self.zmap_fields += ['h_err', 'z_err', 'm_err']
        self.test_data.update({'h_err': 'NaN', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string(self.test_data) in dump)

    def test_ou_hz_error(self):
        """
        Test hz error extraction from origin_uncertainty
        """
        self.zmap_fields += ['h_err', 'z_err', 'm_err']
        self.test_data.update({'h_err': '1.000000', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        o = self.test_event.preferred_origin()
        o.origin_uncertainty.preferred_description = 'horizontal uncertainty'
        o.origin_uncertainty.horizontal_uncertainty = 1.0
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string(self.test_data) in dump)
        # with unsupported preferred_description
        self.test_data.update({'h_err': 'NaN', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        o.origin_uncertainty.preferred_description = 'uncertainty ellipse'
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string(self.test_data) in dump)

    def test_lat_lon_hz_error(self):
        """
        Test hz error extraction from lat/lon
        """
        self.zmap_fields += ['h_err', 'z_err', 'm_err']
        self.test_data.update({'h_err': '0.138679', 'z_err': '0.000000',
                               'm_err': '0.000000'})
        pickler = zmap.Pickler(with_uncertainties=True)
        o = self.test_event.preferred_origin()
        o.latitude_errors.uncertainty = .001
        o.longitude_errors.uncertainty = .001
        dump = pickler.dumps(self.catalog)
        self.assertTrue(self._expected_string(self.test_data) in dump)

    def _expected_string(self, zmap_dict):
        """
        Returns a string as expeced from a zmap dump.

        zmap_dict contains (string) values for all the fields that are expeced
        to have specific values. All other fields are expected to be 'NaN'.
        """
        full_zmap = dict.fromkeys(self.zmap_fields, 'NaN')
        full_zmap.update(zmap_dict)
        string = '\t'.join(full_zmap[f] for f in self.zmap_fields)
        return string


def suite():
    return unittest.makeSuite(ZMAPTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
