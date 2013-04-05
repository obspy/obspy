#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the StationXML reader and writer.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import fnmatch
from io import BytesIO
import inspect
from itertools import izip
import os
import unittest

import obspy
import obspy.station


class StationXMLTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_is_stationxml(self):
        """
        Tests the is_StationXML() function.
        """
        # Check positives.
        stationxmls = [os.path.join(self.data_dir, "minimal_station.xml")]
        for stat in stationxmls:
            self.assertTrue(obspy.station.stationxml.is_StationXML(stat))

        # Check some negatives.
        not_stationxmls = ["Variations-FDSNSXML-SEED.txt",
            "fdsn-station+availability-1.0.xsd", "fdsn-station-1.0.xsd"]
        not_stationxmls = [os.path.join(self.data_dir, os.path.pardir,
            os.path.pardir, "docs", _i) for _i in not_stationxmls]
        for stat in not_stationxmls:
            self.assertFalse(obspy.station.stationxml.is_StationXML(stat))

    def test_read_and_write_minimal_file(self):
        """
        Test that writing the most basic StationXML document possible works.
        """
        filename = os.path.join(self.data_dir, "minimal_station.xml")
        inv = obspy.station.readInventory(filename)

        # Assert the few values that are set directly.
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.networks[0].code, "PY")

        # Write it again. Also validate it to get more confidence. Suppress the
        # writing of the ObsPy related tags to ease testing.
        file_buffer = BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
            _suppress_module_tags=True)
        file_buffer.seek(0, 0)
        new_file = file_buffer.read().splitlines()

        with open(filename, "rb") as fh:
            org_file = fh.read().splitlines()

        self.assertEqual(len(new_file), len(org_file))

        for new_line, org_line in izip(new_file, org_file):
            self.assertEqual(new_line.strip(), org_line.strip())

    def test_writing_module_tags(self):
        """
        Tests the writing of ObsPy related tags.
        """
        net = obspy.station.SeismicNetwork(code="UL")
        inv = obspy.station.SeismicInventory(networks=[net], source="BLU")

        file_buffer = BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True)
        file_buffer.seek(0, 0)
        lines = file_buffer.read().splitlines()
        module_line = [_i.strip() for _i in lines if _i.strip().startswith(
            "<Module>")][0]
        self.assertTrue(fnmatch.fnmatch(module_line,
            "<Module>ObsPy *</Module>"))
        module_URI_line = [_i.strip() for _i in lines if _i.strip().startswith(
            "<ModuleURI>")][0]
        self.assertEqual(module_URI_line,
            "<ModuleURI>http://www.obspy.org</ModuleURI>")

    def test_reading_other_module_tags(self):
        """
        Even though the ObsPy Tags are always written, other tags should be
        able to be read.
        """
        filename = os.path.join(self.data_dir,
            "minimal_with_non_obspy_module_and_sender_tags_station.xml")
        inv = obspy.station.readInventory(filename)
        self.assertEqual(inv.module, "Some Random Module")
        self.assertEqual(inv.module_uri, "http://www.some-random.site")

    def test_reading_and_writing_full_root_tag(self):
        """
        Tests reading and writing a full StationXML root tag.
        """
        filename = os.path.join(self.data_dir,
            "minimal_with_non_obspy_module_and_sender_tags_station.xml")
        inv = obspy.station.readInventory(filename)
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.networks[0].code, "PY")
        self.assertEqual(inv.module, "Some Random Module")
        self.assertEqual(inv.module_uri, "http://www.some-random.site")
        self.assertEqual(inv.sender, "The ObsPy Team")

        # Write it again. Do not write the module tags.
        file_buffer = BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
            _suppress_module_tags=True)
        file_buffer.seek(0, 0)
        lines = [_i.strip() for _i in file_buffer.read().splitlines()]

        with open(filename, "rb") as fh:
            org_lines = fh.readlines()
        # Remove the module tags.
        org_lines = [_i.strip() for _i in org_lines
            if not _i.strip().startswith("<Module")]

        for line, org_line in izip(lines, org_lines):
            self.assertEqual(line, org_line)

    def test_reading_and_writing_full_network_tag(self):
        """
        Tests the reading and writing of a file with a more or less full
        network tag.
        """
        filename = os.path.join(self.data_dir,
            "full_network_field_station.xml")
        inv = obspy.station.readInventory(filename)

        # Assert all the values...
        self.assertEqual(len(inv.networks), 1)
        net = inv.networks[0]
        self.assertEqual(net.code, "PY")
        self.assertEqual(net.start_date, obspy.UTCDateTime(2011, 1, 1))
        self.assertEqual(net.end_date, obspy.UTCDateTime(2012, 1, 1))
        self.assertEqual(net.restricted_status, "open")
        self.assertEqual(net.alternate_code, "PYY")
        self.assertEqual(net.historical_code, "YYP")
        self.assertEqual(net.description, "Some Description...")
        self.assertEqual(len(net.comments), 2)

        comment_1 = net.comments[0]
        self.assertEqual(comment_1.value, "Comment number 1")
        self.assertEqual(comment_1.begin_effective_time,
            obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(comment_1.end_effective_time,
            obspy.UTCDateTime(2008, 2, 3))
        self.assertEqual(len(comment_1.authors), 1)
        authors = comment_1.authors[0]
        self.assertEqual(len(authors.names), 2)
        self.assertEqual(authors.names[0], "This person")
        self.assertEqual(authors.names[1], "has multiple names!")
        self.assertEqual(len(authors.agencies), 3)
        self.assertEqual(authors.agencies[0], "And also")
        self.assertEqual(authors.agencies[1], "many")
        self.assertEqual(authors.agencies[2], "many Agencies")
        self.assertEqual(len(authors.emails), 4)
        self.assertEqual(authors.emails[0], "email1@mail.com")
        self.assertEqual(authors.emails[1], "email2@mail.com")
        self.assertEqual(authors.emails[2], "email3@mail.com")
        self.assertEqual(authors.emails[3], "email4@mail.com")
        self.assertEqual(len(authors.phones), 2)
        self.assertEqual(authors.phones[0].description, "phone number 1")
        self.assertEqual(authors.phones[0].country_code, 49)
        self.assertEqual(authors.phones[0].area_code, 123)
        self.assertEqual(authors.phones[0].phone_number, "456-7890")
        self.assertEqual(authors.phones[1].description, "phone number 2")
        self.assertEqual(authors.phones[1].country_code, 34)
        self.assertEqual(authors.phones[1].area_code, 321)
        self.assertEqual(authors.phones[1].phone_number, "129-7890")

        comment_2 = net.comments[1]
        self.assertEqual(comment_2.value, "Comment number 2")
        self.assertEqual(comment_2.begin_effective_time,
            obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(comment_1.end_effective_time,
            obspy.UTCDateTime(2008, 2, 3))
        self.assertEqual(len(comment_2.authors), 3)
        for _i, author in enumerate(comment_2.authors):
            self.assertEqual(len(author.names), 1)
            self.assertEqual(author.names[0], "Person %i" % (_i + 1))
            self.assertEqual(len(author.agencies), 1)
            self.assertEqual(author.agencies[0], "Some agency")
            self.assertEqual(len(author.emails), 1)
            self.assertEqual(author.emails[0], "email@mail.com")
            self.assertEqual(len(author.phones), 1)
            self.assertEqual(author.phones[0].description, None)
            self.assertEqual(author.phones[0].country_code, 49)
            self.assertEqual(author.phones[0].area_code, 123)
            self.assertEqual(author.phones[0].phone_number, "456-7890")

        # Now write it again and compare to the original file.
        file_buffer = BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
            _suppress_module_tags=True)
        file_buffer.seek(0, 0)
        new_lines = [_i.strip() for _i in file_buffer.read().splitlines()]

        with open(filename, "rb") as open_file:
            org_lines = open_file.read().splitlines()

        # Remove the module lines from the original file.
        org_lines = [_i.strip() for _i in org_lines
            if not _i.strip().startswith("<Module")]

        self.assertEqual(len(new_lines), len(org_lines))
        for new_line, org_line in izip(new_lines, org_lines):
            try:
                self.assertEqual(new_line, org_line)
            except:
                self.assertEqual(sorted(new_line[1:-1].split()),
                    sorted(org_line[1:-1].split()))


def suite():
    return unittest.makeSuite(StationXMLTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
