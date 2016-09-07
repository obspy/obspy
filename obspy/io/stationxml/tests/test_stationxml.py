#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the StationXML reader and writer.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import fnmatch
import inspect
import io
import os
import re
import unittest

import obspy
from obspy.core.inventory import Inventory, Network
import obspy.io.stationxml.core


class StationXMLTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        self.maxDiff = 10000
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def _assert_station_xml_equality(self, xml_file_buffer,
                                     expected_xml_file_buffer):
        """
        Helper function comparing two BytesIO buffers contain Station XML
        files.
        """
        # utf-8 only needed PY2
        new_lines = [_i.decode('utf-8').strip().replace("'", '"')
                     for _i in xml_file_buffer.read().splitlines()]
        # utf-8 only needed PY2
        org_lines = [_i.decode('utf-8').strip().replace("'", '"')
                     for _i in expected_xml_file_buffer.read().splitlines()]

        # Remove the module lines from the original file.
        org_lines = [_i.strip() for _i in org_lines
                     if not _i.strip().startswith("<Module")]

        for new_line, org_line in zip(new_lines, org_lines):
            regex = "<(.*?) (.*?)/?>"

            def callback(pattern):
                part2 = " ".join(sorted(pattern.group(2).split(" ")))
                return "<%s %s>" % (pattern.group(1), part2)

            # resort attributes alphabetically
            org_line = re.sub(regex, callback, org_line, count=1)
            new_line = re.sub(regex, callback, new_line, count=1)
            self.assertEqual(org_line, new_line)

        # Assert the line length at the end to find trailing non-equal lines.
        # If it is done before the line comparison it is oftentimes not very
        # helpful as you do not know which line is missing.
        self.assertEqual(len(new_lines), len(org_lines))

    def test_is_stationxml(self):
        """
        Tests the _is_stationxml() function.
        """
        # Check positives.
        stationxmls = [os.path.join(self.data_dir, "minimal_station.xml")]
        for stat in stationxmls:
            self.assertTrue(obspy.io.stationxml.core._is_stationxml(stat))

        # Check some negatives.
        not_stationxmls = [
            "Variations-FDSNSXML-SEED.txt",
            "fdsn-station+availability-1.0.xsd", "fdsn-station-1.0.xsd"]
        not_stationxmls = [
            os.path.join(self.data_dir, os.path.pardir,
                         os.path.pardir, "data", _i) for _i in not_stationxmls]
        for stat in not_stationxmls:
            self.assertFalse(obspy.io.stationxml.core._is_stationxml(
                stat))

    def test_read_and_write_minimal_file(self):
        """
        Test that writing the most basic StationXML document possible works.
        """
        filename = os.path.join(self.data_dir, "minimal_station.xml")
        inv = obspy.read_inventory(filename)

        # Assert the few values that are set directly.
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.networks[0].code, "PY")

        # Write it again. Also validate it to get more confidence. Suppress the
        # writing of the ObsPy related tags to ease testing.
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(file_buffer,
                                          expected_xml_file_buffer)

    def test_subsecond_read_and_write_minimal_file(self):
        """
        Test reading and writing of sub-second time in datetime field,
        using creation time

        """
        filename = os.path.join(self.data_dir,
                                "minimal_station_with_microseconds.xml")
        inv = obspy.read_inventory(filename)

        # Write it again. Also validate it to get more confidence. Suppress the
        # writing of the ObsPy related tags to ease testing.
        file_buffer = io.BytesIO()

        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(file_buffer,
                                          expected_xml_file_buffer)

    def test_read_and_write_full_file(self):
        """
        Test that reading and writing of a full StationXML document with all
        possible tags works.
        """
        filename = os.path.join(self.data_dir, "full_random_stationxml.xml")
        inv = obspy.read_inventory(filename)

        # Write it again. Also validate it to get more confidence. Suppress the
        # writing of the ObsPy related tags to ease testing.
        file_buffer = io.BytesIO()

        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(file_buffer,
                                          expected_xml_file_buffer)

    def test_writing_module_tags(self):
        """
        Tests the writing of ObsPy related tags.
        """
        net = Network(code="UL")
        inv = Inventory(networks=[net], source="BLU")

        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True)
        file_buffer.seek(0, 0)
        lines = file_buffer.read().decode().splitlines()
        module_line = [_i.strip() for _i in lines if _i.strip().startswith(
            "<Module>")][0]
        self.assertTrue(fnmatch.fnmatch(module_line,
                                        "<Module>ObsPy *</Module>"))
        module_uri_line = [_i.strip() for _i in lines if _i.strip().startswith(
            "<ModuleURI>")][0]
        self.assertEqual(module_uri_line,
                         "<ModuleURI>https://www.obspy.org</ModuleURI>")

    def test_reading_other_module_tags(self):
        """
        Even though the ObsPy Tags are always written, other tags should be
        able to be read.
        """
        filename = os.path.join(
            self.data_dir,
            "minimal_with_non_obspy_module_and_sender_tags_station.xml")
        inv = obspy.read_inventory(filename)
        self.assertEqual(inv.module, "Some Random Module")
        self.assertEqual(inv.module_uri, "http://www.some-random.site")

    def test_reading_and_writing_full_root_tag(self):
        """
        Tests reading and writing a full StationXML root tag.
        """
        filename = os.path.join(
            self.data_dir,
            "minimal_with_non_obspy_module_and_sender_tags_station.xml")
        inv = obspy.read_inventory(filename)
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.networks[0].code, "PY")
        self.assertEqual(inv.module, "Some Random Module")
        self.assertEqual(inv.module_uri, "http://www.some-random.site")
        self.assertEqual(inv.sender, "The ObsPy Team")

        # Write it again. Do not write the module tags.
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(
            file_buffer, expected_xml_file_buffer)

    def test_reading_and_writing_full_network_tag(self):
        """
        Tests the reading and writing of a file with a more or less full
        network tag.
        """
        filename = os.path.join(self.data_dir,
                                "full_network_field_station.xml")
        inv = obspy.read_inventory(filename)

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
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(
            file_buffer,
            expected_xml_file_buffer)

    def test_reading_and_writing_full_station_tag(self):
        """
        Tests the reading and writing of a file with a more or less full
        station tag.
        """
        filename = os.path.join(self.data_dir,
                                "full_station_field_station.xml")
        inv = obspy.read_inventory(filename)

        # Assert all the values...
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv.source, "OBS")
        self.assertEqual(inv.module, "Some Random Module")
        self.assertEqual(inv.module_uri, "http://www.some-random.site")
        self.assertEqual(inv.sender, "The ObsPy Team")
        self.assertEqual(inv.created, obspy.UTCDateTime(2013, 1, 1))
        self.assertEqual(len(inv.networks), 1)
        network = inv.networks[0]
        self.assertEqual(network.code, "PY")

        # Now assert the station specific values.
        self.assertEqual(len(network.stations), 1)
        station = network.stations[0]
        self.assertEqual(station.code, "PY")
        self.assertEqual(station.start_date, obspy.UTCDateTime(2011, 1, 1))
        self.assertEqual(station.end_date, obspy.UTCDateTime(2012, 1, 1))
        self.assertEqual(station.restricted_status, "open")
        self.assertEqual(station.alternate_code, "PYY")
        self.assertEqual(station.historical_code, "YYP")
        self.assertEqual(station.description, "Some Description...")
        self.assertEqual(len(station.comments), 2)
        comment_1 = station.comments[0]
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
        comment_2 = station.comments[1]
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

        self.assertEqual(station.latitude, 10.0)
        self.assertEqual(station.longitude, 20.0)
        self.assertEqual(station.elevation, 100.0)

        self.assertEqual(station.site.name, "Some site")
        self.assertEqual(station.site.description, "Some description")
        self.assertEqual(station.site.town, "Some town")
        self.assertEqual(station.site.county, "Some county")
        self.assertEqual(station.site.region, "Some region")
        self.assertEqual(station.site.country, "Some country")

        self.assertEqual(station.vault, "Some vault")
        self.assertEqual(station.geology, "Some geology")

        self.assertEqual(len(station.equipments), 2)
        self.assertEqual(station.equipments[0].resource_id, "some_id")
        self.assertEqual(station.equipments[0].type, "Some type")
        self.assertEqual(station.equipments[0].description, "Some description")
        self.assertEqual(station.equipments[0].manufacturer,
                         "Some manufacturer")
        self.assertEqual(station.equipments[0].vendor, "Some vendor")
        self.assertEqual(station.equipments[0].model, "Some model")
        self.assertEqual(station.equipments[0].serial_number, "12345-ABC")
        self.assertEqual(station.equipments[0].installation_date,
                         obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(station.equipments[0].removal_date,
                         obspy.UTCDateTime(1999, 5, 5))
        self.assertEqual(station.equipments[0].calibration_dates[0],
                         obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(station.equipments[0].calibration_dates[1],
                         obspy.UTCDateTime(1992, 5, 5))
        self.assertEqual(station.equipments[1].resource_id, "something_new")
        self.assertEqual(station.equipments[1].type, "Some type")
        self.assertEqual(station.equipments[1].description, "Some description")
        self.assertEqual(station.equipments[1].manufacturer,
                         "Some manufacturer")
        self.assertEqual(station.equipments[1].vendor, "Some vendor")
        self.assertEqual(station.equipments[1].model, "Some model")
        self.assertEqual(station.equipments[1].serial_number, "12345-ABC")
        self.assertEqual(station.equipments[1].installation_date,
                         obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(station.equipments[1].removal_date,
                         obspy.UTCDateTime(1999, 5, 5))
        self.assertEqual(station.equipments[1].calibration_dates[0],
                         obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(station.equipments[1].calibration_dates[1],
                         obspy.UTCDateTime(1992, 5, 5))

        self.assertEqual(len(station.operators), 2)
        self.assertEqual(station.operators[0].agencies[0], "Agency 1")
        self.assertEqual(station.operators[0].agencies[1], "Agency 2")
        self.assertEqual(station.operators[0].contacts[0].names[0],
                         "This person")
        self.assertEqual(station.operators[0].contacts[0].names[1],
                         "has multiple names!")
        self.assertEqual(len(station.operators[0].contacts[0].agencies), 3)
        self.assertEqual(station.operators[0].contacts[0].agencies[0],
                         "And also")
        self.assertEqual(station.operators[0].contacts[0].agencies[1], "many")
        self.assertEqual(station.operators[0].contacts[0].agencies[2],
                         "many Agencies")
        self.assertEqual(len(station.operators[0].contacts[0].emails), 4)
        self.assertEqual(station.operators[0].contacts[0].emails[0],
                         "email1@mail.com")
        self.assertEqual(station.operators[0].contacts[0].emails[1],
                         "email2@mail.com")
        self.assertEqual(station.operators[0].contacts[0].emails[2],
                         "email3@mail.com")
        self.assertEqual(station.operators[0].contacts[0].emails[3],
                         "email4@mail.com")
        self.assertEqual(len(station.operators[0].contacts[0].phones), 2)
        self.assertEqual(
            station.operators[0].contacts[0].phones[0].description,
            "phone number 1")
        self.assertEqual(
            station.operators[0].contacts[0].phones[0].country_code, 49)
        self.assertEqual(
            station.operators[0].contacts[0].phones[0].area_code, 123)
        self.assertEqual(
            station.operators[0].contacts[0].phones[0].phone_number,
            "456-7890")
        self.assertEqual(
            station.operators[0].contacts[0].phones[1].description,
            "phone number 2")
        self.assertEqual(
            station.operators[0].contacts[0].phones[1].country_code, 34)
        self.assertEqual(station.operators[0].contacts[0].phones[1].area_code,
                         321)
        self.assertEqual(
            station.operators[0].contacts[0].phones[1].phone_number,
            "129-7890")
        self.assertEqual(station.operators[0].contacts[1].names[0], "Name")
        self.assertEqual(station.operators[0].contacts[1].agencies[0],
                         "Agency")
        self.assertEqual(station.operators[0].contacts[1].emails[0],
                         "email@mail.com")
        self.assertEqual(
            station.operators[0].contacts[1].phones[0].description,
            "phone number 1")
        self.assertEqual(
            station.operators[0].contacts[1].phones[0].country_code, 49)
        self.assertEqual(
            station.operators[0].contacts[1].phones[0].area_code, 123)
        self.assertEqual(
            station.operators[0].contacts[1].phones[0].phone_number,
            "456-7890")
        self.assertEqual(station.operators[0].website, "http://www.web.site")

        self.assertEqual(station.operators[1].agencies[0], "Agency")
        self.assertEqual(station.operators[1].contacts[0].names[0], "New Name")
        self.assertEqual(station.operators[1].contacts[0].agencies[0],
                         "Agency")
        self.assertEqual(station.operators[1].contacts[0].emails[0],
                         "email@mail.com")
        self.assertEqual(
            station.operators[1].contacts[0].phones[0].description,
            "phone number 1")
        self.assertEqual(
            station.operators[1].contacts[0].phones[0].country_code, 49)
        self.assertEqual(station.operators[1].contacts[0].phones[0].area_code,
                         123)
        self.assertEqual(
            station.operators[1].contacts[0].phones[0].phone_number,
            "456-7890")
        self.assertEqual(station.operators[1].website, "http://www.web.site")

        self.assertEqual(station.creation_date, obspy.UTCDateTime(1990, 5, 5))
        self.assertEqual(station.termination_date,
                         obspy.UTCDateTime(2009, 5, 5))
        self.assertEqual(station.total_number_of_channels, 100)
        self.assertEqual(station.selected_number_of_channels, 1)

        self.assertEqual(len(station.external_references), 2)
        self.assertEqual(station.external_references[0].uri,
                         "http://path.to/something")
        self.assertEqual(station.external_references[0].description,
                         "Some description")
        self.assertEqual(station.external_references[1].uri,
                         "http://path.to/something/else")
        self.assertEqual(station.external_references[1].description,
                         "Some other description")

        # Now write it again and compare to the original file.
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", validate=True,
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(file_buffer,
                                          expected_xml_file_buffer)

    def test_reading_and_writing_channel_with_response(self):
        """
        Test the reading and writing of a single channel including a
        multi-stage response object.
        """
        filename = os.path.join(self.data_dir,
                                "IRIS_single_channel_with_response.xml")
        inv = obspy.read_inventory(filename)
        self.assertEqual(inv.source, "IRIS-DMC")
        self.assertEqual(inv.sender, "IRIS-DMC")
        self.assertEqual(inv.created, obspy.UTCDateTime("2013-04-16T06:15:28"))
        # Assert that precisely one channel object has been created.
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(len(inv.networks[0].stations[0].channels), 1)
        network = inv.networks[0]
        station = network.stations[0]
        channel = station.channels[0]
        # Assert some fields of the network. This is extensively tested
        # elsewhere.
        self.assertEqual(network.code, "IU")
        self.assertEqual(network.start_date,
                         obspy.UTCDateTime("1988-01-01T00:00:00"))
        self.assertEqual(network.end_date,
                         obspy.UTCDateTime("2500-12-12T23:59:59"))
        self.assertEqual(network.description,
                         "Global Seismograph Network (GSN - IRIS/USGS)")
        # Assert a few fields of the station. This is extensively tested
        # elsewhere.
        self.assertEqual(station.code, "ANMO")
        self.assertEqual(station.latitude, 34.94591)
        self.assertEqual(station.longitude, -106.4572)
        self.assertEqual(station.elevation, 1820.0)
        self.assertEqual(station.site.name, "Albuquerque, New Mexico, USA")
        # Start to assert the channel reading.
        self.assertEqual(channel.code, "BHZ")
        self.assertEqual(channel.location_code, "10")
        self.assertEqual(channel.start_date,
                         obspy.UTCDateTime("2012-03-13T08:10:00"))
        self.assertEqual(channel.end_date,
                         obspy.UTCDateTime("2599-12-31T23:59:59"))
        self.assertEqual(channel.restricted_status, "open")
        self.assertEqual(channel.latitude, 34.945913)
        self.assertEqual(channel.longitude, -106.457122)
        self.assertEqual(channel.elevation, 1759.0)
        self.assertEqual(channel.depth, 57.0)
        self.assertEqual(channel.azimuth, 0.0)
        self.assertEqual(channel.dip, -90.0)
        self.assertEqual(channel.types, ["CONTINUOUS", "GEOPHYSICAL"])
        self.assertEqual(channel.sample_rate, 40.0)
        self.assertEqual(channel.clock_drift_in_seconds_per_sample, 0.0)
        self.assertEqual(channel.sensor.type,
                         "Guralp CMG3-T Seismometer (borehole)")
        # Check the response.
        response = channel.response
        sensitivity = response.instrument_sensitivity
        self.assertEqual(sensitivity.value, 3.31283E10)
        self.assertEqual(sensitivity.frequency, 0.02)
        self.assertEqual(sensitivity.input_units, "M/S")
        self.assertEqual(sensitivity.input_units_description,
                         "Velocity in Meters Per Second")
        self.assertEqual(sensitivity.output_units, "COUNTS")
        self.assertEqual(sensitivity.output_units_description,
                         "Digital Counts")
        # Assert that there are three stages.
        self.assertEqual(len(response.response_stages), 3)

    def test_stationxml_with_availability(self):
        """
        A variant of StationXML has support for availability information.
        Make sure this works.
        """
        filename = os.path.join(self.data_dir,
                                "stationxml_with_availability.xml")
        inv = obspy.read_inventory(filename, format="stationxml")
        channel = inv[0][0][0]
        self.assertEqual(channel.data_availability.start,
                         obspy.UTCDateTime("1998-10-26T20:35:58"))
        self.assertEqual(channel.data_availability.end,
                         obspy.UTCDateTime("2014-07-21T12:00:00"))

        # Now write it again and compare to the original file.
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML",
                  _suppress_module_tags=True)
        file_buffer.seek(0, 0)

        with open(filename, "rb") as open_file:
            expected_xml_file_buffer = io.BytesIO(open_file.read())
        expected_xml_file_buffer.seek(0, 0)

        self._assert_station_xml_equality(file_buffer,
                                          expected_xml_file_buffer)

    def test_parse_file_with_no_default_namespace(self):
        """
        Tests that reading a file with no default namespace works fine.

        See #1060.
        """
        filename = os.path.join(self.data_dir, "no_default_namespace.xml")
        inv = obspy.read_inventory(filename)
        # Very small file with almost no content.
        self.assertEqual(len(inv.networks), 1)
        self.assertEqual(inv[0].code, "XX")

    def test_numbers_are_written_to_poles_and_zeros(self):
        """
        Poles and zeros have a number attribute. Make sure this is written,
        even if set with a custom complex list.
        """
        # Read default inventory and cut down to a single channel.
        inv = obspy.read_inventory()
        inv.networks = inv[:1]
        inv[0].stations = inv[0][:1]
        inv[0][0].channels = inv[0][0][:1]

        # Manually set the poles and zeros - thus these are cast to our
        # custom classes but number are not yet set.
        inv[0][0][0].response.response_stages[0].poles = [0 + 1j, 2 + 3j]
        inv[0][0][0].response.response_stages[0].zeros = [0 + 1j, 2 + 3j]

        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml", validate=True)
            buf.seek(0, 0)
            data = buf.read().decode()

        # Ugly test - remove all whitespace and make sure the four following
        # lines are part of the written output.
        data = re.sub('\s+', ' ', data)

        self.assertIn(
            '<Zero number="0"> <Real>0.0</Real> '
            '<Imaginary>1.0</Imaginary> </Zero>', data)
        self.assertIn(
            '<Zero number="1"> <Real>2.0</Real> '
            '<Imaginary>3.0</Imaginary> </Zero>', data)
        self.assertIn(
            '<Pole number="0"> <Real>0.0</Real> '
            '<Imaginary>1.0</Imaginary> </Pole>', data)
        self.assertIn(
            '<Pole number="1"> <Real>2.0</Real> '
            '<Imaginary>3.0</Imaginary> </Pole>', data)


def suite():
    return unittest.makeSuite(StationXMLTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
