# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import datetime
import os
import unittest
import warnings

from obspy import read_events, read_inventory
from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.io.shapefile.core import _write_shapefile, HAS_PYSHP

if HAS_PYSHP:
    import shapefile


SHAPEFILE_SUFFIXES = (".shp", ".shx", ".dbf", ".prj")
expected_catalog_fields = [
    ('DeletionFlag', 'C', 1, 0),
    ['EventID', 'C', 100, 0],
    ['OriginID', 'C', 100, 0],
    ['MagID', 'C', 100, 0],
    ['Date', 'D', 8, 0],
    ['OriginTime', 'N', 20, 6],
    ['FirstPick', 'N', 20, 6],
    ['Longitude', 'N', 16, 10],
    ['Latitude', 'N', 16, 10],
    ['Depth', 'N', 8, 3],
    ['MinHorUncM', 'N', 12, 3],
    ['MaxHorUncM', 'N', 12, 3],
    ['MaxHorAzi', 'N', 7, 3],
    ['OriUncDesc', 'C', 40, 0],
    ['Magnitude', 'N', 8, 3]]
expected_catalog_records = [
    ['quakeml:us.anss.org/event/20120101052755.98',
     'quakeml:us.anss.org/origin/20120101052755.98',
     'quakeml:us.anss.org/magnitude/20120101052755.98/mb',
     datetime.date(2012, 1, 1),
     1325395675.98,
     1325395728.18,
     138.072,
     31.456,
     365.3,
     None,
     None,
     None,
     'None',
     6.2],
    ['smi:local/0eee2e6f-064b-458a-934f-c5d3105e9529',
     'smi:local/a2260002-95c6-42f7-8c44-f46124355228',
     'None',
     datetime.date(2006, 7, 15),
     1152984080.19567,
     1152984080.63,
     7.736781,
     51.657659,
     1.434,
     1136.0,
     1727.42,
     69.859,
     'uncertainty ellipse',
     None]]
expected_inventory_fields = [
    ('DeletionFlag', 'C', 1, 0),
    ['Network', 'C', 20, 0],
    ['Station', 'C', 20, 0],
    ['Longitude', 'N', 16, 10],
    ['Latitude', 'N', 16, 10],
    ['Elevation', 'N', 9, 3],
    ['StartDate', 'D', 8, 0],
    ['EndDate', 'D', 8, 0],
    ['Channels', 'C', 254, 0]]
expected_inventory_records = [
    ['GR', 'FUR', 11.2752, 48.162899, 565.0, datetime.date(2006, 12, 16),
     None, '.HHZ,.HHN,.HHE,.BHZ,.BHN,.BHE,.LHZ,.LHN,.LHE,.VHZ,.VHN,.VHE'],
    ['GR', 'WET', 12.8782, 49.144001, 613.0, datetime.date(2007, 2, 2), None,
     '.HHZ,.HHN,.HHE,.BHZ,.BHN,.BHE,.LHZ,.LHN,.LHE'],
    ['BW', 'RJOB', 12.795714, 47.737167, 860.0, datetime.date(2001, 5, 15),
     datetime.date(2006, 12, 12), '.EHZ,.EHN,.EHE'],
    ['BW', 'RJOB', 12.795714, 47.737167, 860.0, datetime.date(2006, 12, 13),
     datetime.date(2007, 12, 17), '.EHZ,.EHN,.EHE'],
    ['BW', 'RJOB', 12.795714, 47.737167, 860.0, datetime.date(2007, 12, 17),
     None, '.EHZ,.EHN,.EHE']]


def _close_shapefile_reader(reader):
    """
    Current pyshp version 1.2.12 doesn't properly close files, so for now do
    this manually during tests. (see GeospatialPython/pyshp#107)
    """
    for key in ('dbf', 'shx', 'shp'):
        attribute = getattr(reader, key, None)
        if attribute is not None:
            try:
                attribute.close()
            except (AttributeError, IOError):
                pass


@unittest.skipIf(not HAS_PYSHP, 'pyshp not installed')
class ShapefileTestCase(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data')
        self.catalog_shape_basename = os.path.join(self.path, 'catalog')
        self.inventory_shape_basename = os.path.join(self.path, 'inventory')

    def test_write_catalog_shapefile(self):
        # read two events with uncertainties, one deserializes with "confidence
        # ellipsoid" origin uncertainty which is not yet implemented for
        # shapefile output and should show a warning
        cat = read_events('/path/to/mchedr.dat')
        cat += read_events('/path/to/nlloc.qml')
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always')
                _write_shapefile(cat, "catalog.shp")
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message),
                        'Encountered an event with origin uncertainty '
                        'description of type "confidence ellipsoid". This is '
                        'not yet implemented for output as shapefile. No '
                        'origin uncertainty will be added to shapefile for '
                        'such events.')
                except:
                    continue
                break
            else:
                raise
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("catalog" + suffix))
            shp = shapefile.Reader("catalog.shp")
            # check contents of shapefile that we just wrote
            self.assertEqual(shp.fields, expected_catalog_fields)
            self.assertEqual(shp.records(), expected_catalog_records)
            self.assertEqual(shp.shapeType, shapefile.POINT)
            _close_shapefile_reader(shp)

    def test_write_catalog_shapefile_via_plugin(self):
        # read two events with uncertainties, one deserializes with "confidence
        # ellipsoid" origin uncertainty which is not yet implemented for
        # shapefile output and should show a warning
        cat = read_events('/path/to/mchedr.dat')
        cat += read_events('/path/to/nlloc.qml')
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always')
                cat.write("catalog.shp", "SHAPEFILE")
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message),
                        'Encountered an event with origin uncertainty '
                        'description of type "confidence ellipsoid". This is '
                        'not yet implemented for output as shapefile. No '
                        'origin uncertainty will be added to shapefile for '
                        'such events.')
                except:
                    continue
                break
            else:
                raise
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("catalog" + suffix))
            shp = shapefile.Reader("catalog.shp")
            # check contents of shapefile that we just wrote
            self.assertEqual(shp.fields, expected_catalog_fields)
            self.assertEqual(shp.records(), expected_catalog_records)
            self.assertEqual(shp.shapeType, shapefile.POINT)
            _close_shapefile_reader(shp)

    def test_write_inventory_shapefile(self):
        inv = read_inventory()
        with TemporaryWorkingDirectory():
            _write_shapefile(inv, "inventory.shp")
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("inventory" + suffix))
            shp = shapefile.Reader("inventory.shp")
            # check contents of shapefile that we just wrote
            self.assertEqual(shp.fields, expected_inventory_fields)
            self.assertEqual(shp.records(), expected_inventory_records)
            self.assertEqual(shp.shapeType, shapefile.POINT)
            _close_shapefile_reader(shp)

    def test_write_inventory_shapefile_via_plugin(self):
        inv = read_inventory()
        with TemporaryWorkingDirectory():
            inv.write("inventory.shp", "SHAPEFILE")
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("inventory" + suffix))
            shp = shapefile.Reader("inventory.shp")
            # check contents of shapefile that we just wrote
            self.assertEqual(shp.fields, expected_inventory_fields)
            self.assertEqual(shp.records(), expected_inventory_records)
            self.assertEqual(shp.shapeType, shapefile.POINT)
            _close_shapefile_reader(shp)


def suite():
    return unittest.makeSuite(ShapefileTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
