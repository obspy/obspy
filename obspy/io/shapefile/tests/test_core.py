# -*- coding: utf-8 -*-
import copy
import datetime
import os
import unittest
import warnings

from obspy import read_events, read_inventory
from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.io.shapefile.core import (
    _write_shapefile, HAS_PYSHP, PYSHP_VERSION_AT_LEAST_1_2_11,
    PYSHP_VERSION_WARNING)

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
# set up expected results with extra 'Region' field
expected_catalog_fields_with_region = copy.deepcopy(expected_catalog_fields)
expected_catalog_fields_with_region.append(['Region', 'C', 50, 0])
expected_catalog_records_with_region = copy.deepcopy(expected_catalog_records)
expected_catalog_records_with_region[0].append('SOUTHEAST OF HONSHU, JAPAN')
expected_catalog_records_with_region[1].append('GERMANY')
# set up expected results with extra 'Comment' field
expected_inventory_fields_with_comment = copy.deepcopy(
    expected_inventory_fields)
expected_inventory_fields_with_comment.append(['Comment', 'C', 50, 0])
expected_inventory_records_with_comment = copy.deepcopy(
    expected_inventory_records)
expected_inventory_records_with_comment[0].append('Abc')
expected_inventory_records_with_comment[1].append(None)
expected_inventory_records_with_comment[2].append('123')
expected_inventory_records_with_comment[3].append('Some comment')
expected_inventory_records_with_comment[4].append('')


def _assert_records_and_fields(got_fields, got_records, expected_fields,
                               expected_records):
    null_values = {'None', '', None}

    if got_fields != expected_fields:
        msg = 'Expected Fields:\n{!s}\nActual Fields\n{!s}'
        msg = msg.format(expected_fields, got_fields)
        raise AssertionError(msg)
    if len(got_records) != len(expected_records):
        msg = 'Expected Fields:\n{!s}\nActual Fields\n{!s}'
        msg = msg.format(expected_fields, got_fields)
        raise AssertionError(msg)
    # omit first field which is deletion flag field
    for got_record, expected_record in zip(got_records, expected_records):
        for i, (field, got, expected) in enumerate(zip(
                expected_fields[1:], got_record, expected_record)):
            # on older pyshp <=1.2.10 date fields don't get cast to
            # datetime.date on reading..
            field_type = field[1]
            if not PYSHP_VERSION_AT_LEAST_1_2_11:
                if field_type == 'D':
                    if got == 'None':
                        got = None
                    # yet another workaround for testing results from old pyshp
                    # Reader.. seem to sometimes return the raw unconverted
                    # string e.g. '20120101'
                    elif len(got) == 8:
                        got = datetime.date(
                            year=int(got[:4]),
                            month=int(got[4:6].lstrip('0')),
                            day=int(got[6:8].lstrip('0')))
                    else:
                        year, month, day = got
                        got = datetime.date(year=year, month=month, day=day)
                # it seems on older pyshp version empty numeric fields don't
                # get cast to None properly during reading..
                elif field_type == 'N':
                    if got is None and expected is None:
                        continue
                    elif hasattr(got, 'strip') and not got.strip(b' '):
                        got = None
                    else:
                        # old pyshp is seriously buggy and doesn't respect the
                        # specified precision when writing numerical fields
                        if round(got, 1) == round(expected, 1):
                            continue
            if got != expected:
                # pyshp 2.0.0 now seems to write empty str rather than 'None'
                # try to just check for null-ish values and exit if found
                if got in null_values and expected in null_values:
                    return
                msg = "Record {} mismatching:\nExpected: '{!s}'\nGot: '{!s}'"
                msg = msg.format(i, expected, got)
                raise AssertionError(msg)


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
                except AssertionError:
                    continue
                break
            else:
                raise Exception
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("catalog" + suffix))
            with open("catalog.shp", "rb") as fh_shp, \
                    open("catalog.dbf", "rb") as fh_dbf, \
                    open("catalog.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_catalog_fields,
                    expected_records=expected_catalog_records)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)

    def test_write_catalog_shapefile_with_extra_field(self):
        """
        Tests writing a catalog with an additional custom database column
        """
        cat = read_events('/path/to/mchedr.dat')
        cat += read_events('/path/to/nlloc.qml')
        extra_fields = [('Region', 'C', 50, None,
                        ['SOUTHEAST OF HONSHU, JAPAN', 'GERMANY'])]
        bad_extra_fields_wrong_length = [('Region', 'C', 50, None, ['ABC'])]
        bad_extra_fields_name_clash = [('Magnitude', 'C', 50, None, ['ABC'])]

        with TemporaryWorkingDirectory():
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always')
                # test some bad calls that should raise an Exception
                with self.assertRaises(ValueError) as cm:
                    _write_shapefile(
                        cat, "catalog.shp",
                        extra_fields=bad_extra_fields_wrong_length)
                self.assertEqual(
                    str(cm.exception), "list of values for each item in "
                    "'extra_fields' must have same length as Catalog object")
                with self.assertRaises(ValueError) as cm:
                    _write_shapefile(
                        cat, "catalog.shp",
                        extra_fields=bad_extra_fields_name_clash)
                self.assertEqual(
                    str(cm.exception), "Conflict with existing field named "
                    "'Magnitude'.")
                # now test a good call that should work
                _write_shapefile(cat, "catalog.shp", extra_fields=extra_fields)
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message),
                        'Encountered an event with origin uncertainty '
                        'description of type "confidence ellipsoid". This is '
                        'not yet implemented for output as shapefile. No '
                        'origin uncertainty will be added to shapefile for '
                        'such events.')
                except AssertionError:
                    continue
                break
            else:
                raise Exception
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("catalog" + suffix))
            with open("catalog.shp", "rb") as fh_shp, \
                    open("catalog.dbf", "rb") as fh_dbf, \
                    open("catalog.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_catalog_fields_with_region,
                    expected_records=expected_catalog_records_with_region)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)
            # For some reason, on windows the files are still in use when
            # TemporaryWorkingDirectory tries to remove the directory.
            self.assertTrue(fh_shp.closed)
            self.assertTrue(fh_dbf.closed)
            self.assertTrue(fh_shx.closed)

    def test_write_inventory_shapefile_with_extra_field(self):
        """
        Tests writing an inventory with an additional custom database column
        """
        inv = read_inventory()
        extra_fields = [('Comment', 'C', 50, None,
                        ['Abc', None, '123', 'Some comment', ''])]
        bad_extra_fields_wrong_length = [('Comment', 'C', 50, None, ['ABC'])]
        bad_extra_fields_name_clash = [('Station', 'C', 50, None, ['ABC'])]

        with TemporaryWorkingDirectory():
            # test some bad calls that should raise an Exception
            with self.assertRaises(ValueError) as cm:
                _write_shapefile(
                    inv, "inventory.shp",
                    extra_fields=bad_extra_fields_wrong_length)
            self.assertEqual(
                str(cm.exception), "list of values for each item in "
                "'extra_fields' must have same length as the count of all "
                "Stations combined across all Networks.")
            with self.assertRaises(ValueError) as cm:
                _write_shapefile(
                    inv, "inventory.shp",
                    extra_fields=bad_extra_fields_name_clash)
            self.assertEqual(
                str(cm.exception), "Conflict with existing field named "
                "'Station'.")
            # now test a good call that should work
            _write_shapefile(inv, "inventory.shp", extra_fields=extra_fields)
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("inventory" + suffix))
            with open("inventory.shp", "rb") as fh_shp, \
                    open("inventory.dbf", "rb") as fh_dbf, \
                    open("inventory.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_inventory_fields_with_comment,
                    expected_records=expected_inventory_records_with_comment)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)
            # For some reason, on windows the files are still in use when
            # TemporaryWorkingDirectory tries to remove the directory.
            self.assertTrue(fh_shp.closed)
            self.assertTrue(fh_dbf.closed)
            self.assertTrue(fh_shx.closed)

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
                except AssertionError:
                    continue
                break
            else:
                raise Exception
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("catalog" + suffix))
            with open("catalog.shp", "rb") as fh_shp, \
                    open("catalog.dbf", "rb") as fh_dbf, \
                    open("catalog.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_catalog_fields,
                    expected_records=expected_catalog_records)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)

    def test_write_inventory_shapefile(self):
        inv = read_inventory()
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always')
                _write_shapefile(inv, "inventory.shp")
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message), PYSHP_VERSION_WARNING)
                except AssertionError:
                    continue
                break
            else:
                if not PYSHP_VERSION_AT_LEAST_1_2_11:
                    raise AssertionError('pyshape version warning not shown')
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("inventory" + suffix))
            with open("inventory.shp", "rb") as fh_shp, \
                    open("inventory.dbf", "rb") as fh_dbf, \
                    open("inventory.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_inventory_fields,
                    expected_records=expected_inventory_records)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)

    def test_write_inventory_shapefile_via_plugin(self):
        inv = read_inventory()
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always')
                inv.write("inventory.shp", "SHAPEFILE")
            for w_ in w:
                try:
                    self.assertEqual(
                        str(w_.message), PYSHP_VERSION_WARNING)
                except AssertionError:
                    continue
                break
            else:
                if not PYSHP_VERSION_AT_LEAST_1_2_11:
                    raise AssertionError('pyshape version warning not shown')
            for suffix in SHAPEFILE_SUFFIXES:
                self.assertTrue(os.path.isfile("inventory" + suffix))
            with open("inventory.shp", "rb") as fh_shp, \
                    open("inventory.dbf", "rb") as fh_dbf, \
                    open("inventory.shx", "rb") as fh_shx:
                shp = shapefile.Reader(shp=fh_shp, shx=fh_shx, dbf=fh_dbf)
                # check contents of shapefile that we just wrote
                _assert_records_and_fields(
                    got_fields=shp.fields, got_records=shp.records(),
                    expected_fields=expected_inventory_fields,
                    expected_records=expected_inventory_records)
                self.assertEqual(shp.shapeType, shapefile.POINT)
                _close_shapefile_reader(shp)
