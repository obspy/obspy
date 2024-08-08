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
import fnmatch
import io
import re
import warnings

from lxml import etree

import obspy
import obspy.io.stationxml.core
from obspy import UTCDateTime
from obspy.core.util import AttribDict, CatchAndAssertWarnings
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.inventory import (Inventory, Network, ResponseStage)
from obspy.core.inventory.util import DataAvailability
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.stationxml.core import _read_stationxml
import pytest


class TestStationXML():
    """
    """
    @classmethod
    def setup_class(cls):
        cls.maxDiff = 10000

    def _assert_station_xml_equality(self, xml_file_buffer,
                                     expected_xml_file_buffer):
        """
        Helper function comparing two BytesIO buffers contain Station XML
        files.
        """
        new_lines = [_i.decode('utf-8').strip().replace("'", '"')
                     for _i in xml_file_buffer.read().splitlines()]
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
            assert org_line == new_line

        # Assert the line length at the end to find trailing non-equal lines.
        # If it is done before the line comparison it is oftentimes not very
        # helpful as you do not know which line is missing.
        assert len(new_lines) == len(org_lines)

    def test_is_stationxml(self, testdata, datapath):
        """
        Tests the _is_stationxml() function.
        """
        # Check positives.
        stationxmls = [testdata["minimal_station.xml"]]
        for stat in stationxmls:
            assert obspy.io.stationxml.core._is_stationxml(stat)

        # Check some negatives.
        not_stationxmls = [
            "Variations-FDSNSXML-SEED.txt",
            "fdsn-station+availability-1.0.xsd", "fdsn-station-1.0.xsd"]
        not_stationxmls = [datapath.parent.parent / 'data' / name
                           for name in not_stationxmls]
        for stat in not_stationxmls:
            assert not obspy.io.stationxml.core._is_stationxml(stat)

    def test_different_write_levels(self, testdata):
        """
        Tests different levels of writing
        """
        filename = testdata["stationxml_BK.CMB.__.LKS.xml"]
        inv = obspy.read_inventory(filename)

        # Write to network level
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", level="network")
        file_buffer.seek(0, 0)

        network_inv = obspy.read_inventory(file_buffer)

        assert len(network_inv.networks) == len(inv.networks)

        for net in network_inv.networks:
            assert len(net.stations) == 0

        # Write to station level
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", level="station")
        file_buffer.seek(0, 0)

        station_inv = obspy.read_inventory(file_buffer)

        for net in station_inv.networks:
            assert len(net.stations) == len(inv[0].stations)
            for sta in net.stations:
                assert len(sta.channels) == 0

        # Write to channel level
        file_buffer = io.BytesIO()
        inv.write(file_buffer, format="StationXML", level="channel")
        file_buffer.seek(0, 0)

        channel_inv = obspy.read_inventory(file_buffer)

        for net in channel_inv.networks:
            assert len(net.stations) == len(inv[0].stations)
            for sta in net.stations:
                assert len(sta.channels) == len(inv[0][0].channels)
                for cha in sta.channels:
                    assert cha.response is None

    def test_read_and_write_minimal_file(self, testdata):
        """
        Test that writing the most basic StationXML document possible works.
        """
        filename = testdata["minimal_station.xml"]
        inv = obspy.read_inventory(filename)

        # Assert the few values that are set directly.
        assert inv.source == "OBS"
        assert inv.created == obspy.UTCDateTime(2013, 1, 1)
        assert len(inv.networks) == 1
        assert inv.networks[0].code == "PY"

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

    def test_subsecond_read_and_write_minimal_file(self, testdata):
        """
        Test reading and writing of sub-second time in datetime field,
        using creation time

        """
        filename = testdata["minimal_station_with_microseconds.xml"]
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

    def test_read_and_write_full_file(self, testdata):
        """
        Test that reading and writing of a full StationXML document with all
        possible tags works.
        """
        filename = testdata["full_random_stationxml.xml"]
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

        # test some new fields added in StationXML 1.1 specifically
        net = inv[0]
        sta = net[0]
        cha = sta[0]
        assert len(net.identifiers) == 2
        assert net.identifiers[0] == "abc:def"
        assert net.identifiers[1] == "uvw:xyz"
        assert len(net.operators) == 2
        assert net.operators[0].agency == 'ABC0.oszQNsC4l66ieQFM'
        # accessing "agencies" this should raise a DeprecationWarning but its
        # tested in another test case already
        with CatchAndAssertWarnings():
            assert net.operators[0].agencies == \
                             ['ABC0.oszQNsC4l66ieQFM']
        assert net.operators[0].contacts[0].names[0] == 'A'
        # check WaterLevel tag
        assert sta.water_level == 250.4
        assert sta.water_level.lower_uncertainty == 2.3
        assert sta.water_level.upper_uncertainty == 4.2
        assert sta.water_level.unit == 'METERS'
        assert cha.water_level == 631.2
        assert cha.water_level.lower_uncertainty == 5.3
        assert cha.water_level.upper_uncertainty == 3.2
        assert cha.water_level.unit == 'METERS'
        # multiple equipments allowed now on channel, deprecation for old
        # single equipment attribute
        assert len(cha.equipments) == 2
        assert cha.equipments[1].type == "some type"
        msg = (r"Attribute 'equipment' \(holding a single Equipment\) is "
               r"deprecated in favor of 'equipments' which now holds a list "
               r"of Equipment objects \(following changes in StationXML 1.1\) "
               r"and might be removed in the future. Returning the first "
               r"entry found in 'equipments'.")
        with CatchAndAssertWarnings(
                clear=['obspy.core.inventory.channel'],
                expected=[(ObsPyDeprecationWarning, msg)]):
            assert cha.equipment.type == cha.equipments[0].type
        # check new measurementMethod attributes
        assert sta.latitude.measurement_method == "GPS"
        assert sta.longitude.measurement_method == "GPS"
        assert sta.elevation.measurement_method == "digital elevation model"
        assert cha.azimuth.measurement_method == "fibre optic gyro compass"
        # check data availability tags
        assert net.data_availability.start == UTCDateTime(2011, 2, 3, 4, 5, 6)
        assert net.data_availability.end == UTCDateTime(2011, 3, 4, 5, 6, 7)
        assert len(net.data_availability.spans) == 2
        span1 = net.data_availability.spans[0]
        span2 = net.data_availability.spans[1]
        assert span1.start == UTCDateTime(2012, 2, 3, 4, 5, 6)
        assert span1.end == UTCDateTime(2012, 3, 4, 5, 6, 7)
        assert span1.number_of_segments == 5
        assert span1.maximum_time_tear == 7.8
        assert span2.start == UTCDateTime(2013, 2, 3, 4, 5, 6)
        assert span2.end == UTCDateTime(2013, 3, 4, 5, 6, 7)
        assert span2.number_of_segments == 8
        assert span2.maximum_time_tear == 2.4
        # test sourceID
        assert net.source_id == "http://www.example.com"
        assert sta.source_id == "http://www.example2.com"
        assert cha.source_id == "http://www.example3.com"
        # Comment topic
        assert net.comments[0].subject == "my topic"
        # Comment id optional now
        assert net.comments[0].id is None
        # storage_format was deprecated since it was removed in StationXML 1.1
        msg = (r"Attribute 'storage_format' was removed in accordance with "
               r"StationXML 1\.1, ignoring\.")
        with CatchAndAssertWarnings(
                clear=['obspy.core.inventory.channel'],
                expected=[(ObsPyDeprecationWarning, msg)]):
            cha.storage_format = "something"
        msg = (r"Attribute 'storage_format' was removed in accordance with "
               r"StationXML 1\.1, returning None\.")
        with CatchAndAssertWarnings(
                clear=['obspy.core.inventory.channel'],
                expected=[(ObsPyDeprecationWarning, msg)]):
            assert cha.storage_format is None
        # check new number attributes on Numerator/Denominator
        resp = cha.response
        stage = resp.response_stages[1]
        assert stage.numerator[0].number == 1
        assert stage.numerator[1].number == 2
        assert stage.denominator[0].number == 3
        assert stage.denominator[1].number == 4

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
        assert fnmatch.fnmatch(module_line, "<Module>ObsPy *</Module>")
        module_uri_line = [_i.strip() for _i in lines if
                           _i.strip().startswith("<ModuleURI>")][0]
        assert module_uri_line == \
            "<ModuleURI>https://www.obspy.org</ModuleURI>"

    def test_reading_other_module_tags(self, testdata):
        """
        Even though the ObsPy Tags are always written, other tags should be
        able to be read.
        """
        filename = testdata[
            "minimal_with_non_obspy_module_and_sender_tags_station.xml"]
        inv = obspy.read_inventory(filename)
        assert inv.module == "Some Random Module"
        assert inv.module_uri == "http://www.some-random.site"

    def test_reading_and_writing_full_root_tag(self, testdata):
        """
        Tests reading and writing a full StationXML root tag.
        """
        filename = testdata[
            "minimal_with_non_obspy_module_and_sender_tags_station.xml"]
        inv = obspy.read_inventory(filename)
        assert inv.source == "OBS"
        assert inv.created == obspy.UTCDateTime(2013, 1, 1)
        assert len(inv.networks) == 1
        assert inv.networks[0].code == "PY"
        assert inv.module == "Some Random Module"
        assert inv.module_uri == "http://www.some-random.site"
        assert inv.sender == "The ObsPy Team"

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

    def test_reading_and_writing_full_network_tag(self, testdata):
        """
        Tests the reading and writing of a file with a more or less full
        network tag.
        """
        filename = testdata["full_network_field_station.xml"]
        inv = obspy.read_inventory(filename)

        # Assert all the values...
        assert len(inv.networks) == 1
        net = inv.networks[0]
        assert net.code == "PY"
        assert net.start_date == obspy.UTCDateTime(2011, 1, 1)
        assert net.end_date == obspy.UTCDateTime(2012, 1, 1)
        assert net.restricted_status == "open"
        assert net.alternate_code == "PYY"
        assert net.historical_code == "YYP"
        assert net.description == "Some Description..."
        assert len(net.comments) == 2

        comment_1 = net.comments[0]
        assert comment_1.value == "Comment number 1"
        assert comment_1.begin_effective_time == obspy.UTCDateTime(1990, 5, 5)
        assert comment_1.end_effective_time == obspy.UTCDateTime(2008, 2, 3)
        assert len(comment_1.authors) == 1
        authors = comment_1.authors[0]
        assert len(authors.names) == 2
        assert authors.names[0] == "This person"
        assert authors.names[1] == "has multiple names!"
        assert len(authors.agencies) == 3
        assert authors.agencies[0] == "And also"
        assert authors.agencies[1] == "many"
        assert authors.agencies[2] == "many Agencies"
        assert len(authors.emails) == 4
        assert authors.emails[0] == "email1@mail.com"
        assert authors.emails[1] == "email2@mail.com"
        assert authors.emails[2] == "email3@mail.com"
        assert authors.emails[3] == "email4@mail.com"
        assert len(authors.phones) == 2
        assert authors.phones[0].description == "phone number 1"
        assert authors.phones[0].country_code == 49
        assert authors.phones[0].area_code == 123
        assert authors.phones[0].phone_number == "456-7890"
        assert authors.phones[1].description == "phone number 2"
        assert authors.phones[1].country_code == 34
        assert authors.phones[1].area_code == 321
        assert authors.phones[1].phone_number == "129-7890"

        comment_2 = net.comments[1]
        assert comment_2.value == "Comment number 2"
        assert comment_2.begin_effective_time == obspy.UTCDateTime(1990, 5, 5)
        assert comment_1.end_effective_time == obspy.UTCDateTime(2008, 2, 3)
        assert len(comment_2.authors) == 3
        for _i, author in enumerate(comment_2.authors):
            assert len(author.names) == 1
            assert author.names[0] == "Person %i" % (_i + 1)
            assert len(author.agencies) == 1
            assert author.agencies[0] == "Some agency"
            assert len(author.emails) == 1
            assert author.emails[0] == "email@mail.com"
            assert len(author.phones) == 1
            assert author.phones[0].description is None
            assert author.phones[0].country_code == 49
            assert author.phones[0].area_code == 123
            assert author.phones[0].phone_number == "456-7890"

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

    def test_reading_and_writing_full_station_tag(self, testdata):
        """
        Tests the reading and writing of a file with a more or less full
        station tag.
        """
        filename = testdata["full_station_field_station.xml"]
        inv = obspy.read_inventory(filename)

        # Assert all the values...
        assert len(inv.networks) == 1
        assert inv.source == "OBS"
        assert inv.module == "Some Random Module"
        assert inv.module_uri == "http://www.some-random.site"
        assert inv.sender == "The ObsPy Team"
        assert inv.created == obspy.UTCDateTime(2013, 1, 1)
        assert len(inv.networks) == 1
        network = inv.networks[0]
        assert network.code == "PY"

        # Now assert the station specific values.
        assert len(network.stations) == 1
        station = network.stations[0]
        assert station.code == "PY"
        assert station.start_date == obspy.UTCDateTime(2011, 1, 1)
        assert station.end_date == obspy.UTCDateTime(2012, 1, 1)
        assert station.restricted_status == "open"
        assert station.alternate_code == "PYY"
        assert station.historical_code == "YYP"
        assert station.description == "Some Description..."
        assert len(station.comments) == 2
        comment_1 = station.comments[0]
        assert comment_1.value == "Comment number 1"
        assert comment_1.begin_effective_time == obspy.UTCDateTime(1990, 5, 5)
        assert comment_1.end_effective_time == obspy.UTCDateTime(2008, 2, 3)
        assert len(comment_1.authors) == 1
        authors = comment_1.authors[0]
        assert len(authors.names) == 2
        assert authors.names[0] == "This person"
        assert authors.names[1] == "has multiple names!"
        assert len(authors.agencies) == 3
        assert authors.agencies[0] == "And also"
        assert authors.agencies[1] == "many"
        assert authors.agencies[2] == "many Agencies"
        assert len(authors.emails) == 4
        assert authors.emails[0] == "email1@mail.com"
        assert authors.emails[1] == "email2@mail.com"
        assert authors.emails[2] == "email3@mail.com"
        assert authors.emails[3] == "email4@mail.com"
        assert len(authors.phones) == 2
        assert authors.phones[0].description == "phone number 1"
        assert authors.phones[0].country_code == 49
        assert authors.phones[0].area_code == 123
        assert authors.phones[0].phone_number == "456-7890"
        assert authors.phones[1].description == "phone number 2"
        assert authors.phones[1].country_code == 34
        assert authors.phones[1].area_code == 321
        assert authors.phones[1].phone_number == "129-7890"
        comment_2 = station.comments[1]
        assert comment_2.value == "Comment number 2"
        assert comment_2.begin_effective_time == obspy.UTCDateTime(1990, 5, 5)
        assert comment_1.end_effective_time == obspy.UTCDateTime(2008, 2, 3)
        assert len(comment_2.authors) == 3
        for _i, author in enumerate(comment_2.authors):
            assert len(author.names) == 1
            assert author.names[0] == "Person %i" % (_i + 1)
            assert len(author.agencies) == 1
            assert author.agencies[0] == "Some agency"
            assert len(author.emails) == 1
            assert author.emails[0] == "email@mail.com"
            assert len(author.phones) == 1
            assert author.phones[0].description is None
            assert author.phones[0].country_code == 49
            assert author.phones[0].area_code == 123
            assert author.phones[0].phone_number == "456-7890"

        assert station.latitude == 10.0
        assert station.longitude == 20.0
        assert station.elevation == 100.0

        assert station.site.name == "Some site"
        assert station.site.description == "Some description"
        assert station.site.town == "Some town"
        assert station.site.county == "Some county"
        assert station.site.region == "Some region"
        assert station.site.country == "Some country"

        assert station.vault == "Some vault"
        assert station.geology == "Some geology"

        assert len(station.equipments) == 2
        assert station.equipments[0].resource_id == "some_id"
        assert station.equipments[0].type == "Some type"
        assert station.equipments[0].description == "Some description"
        assert station.equipments[0].manufacturer == "Some manufacturer"
        assert station.equipments[0].vendor == "Some vendor"
        assert station.equipments[0].model == "Some model"
        assert station.equipments[0].serial_number == "12345-ABC"
        assert station.equipments[0].installation_date == \
            obspy.UTCDateTime(1990, 5, 5)
        assert station.equipments[0].removal_date == \
            obspy.UTCDateTime(1999, 5, 5)
        assert station.equipments[0].calibration_dates[0] == \
            obspy.UTCDateTime(1990, 5, 5)
        assert station.equipments[0].calibration_dates[1] == \
            obspy.UTCDateTime(1992, 5, 5)
        assert station.equipments[1].resource_id == "something_new"
        assert station.equipments[1].type == "Some type"
        assert station.equipments[1].description == "Some description"
        assert station.equipments[1].manufacturer == "Some manufacturer"
        assert station.equipments[1].vendor == "Some vendor"
        assert station.equipments[1].model == "Some model"
        assert station.equipments[1].serial_number == "12345-ABC"
        assert station.equipments[1].installation_date == \
            obspy.UTCDateTime(1990, 5, 5)
        assert station.equipments[1].removal_date == \
            obspy.UTCDateTime(1999, 5, 5)
        assert station.equipments[1].calibration_dates[0] == \
            obspy.UTCDateTime(1990, 5, 5)
        assert station.equipments[1].calibration_dates[1] == \
            obspy.UTCDateTime(1992, 5, 5)

        assert len(station.operators) == 2
        assert station.operators[0].agency == "Agency 1"
        # legacy, "agencies" was a thing for StationXML 1.0
        regex = (
            r"Attribute 'agencies' \(holding a list of strings as Agencies\) "
            r"is deprecated in favor of 'agency' which now holds a single "
            r"string \(following changes in StationXML 1\.1\) and might be "
            r"removed in the future. Returning a list built up of the single "
            r"agency or an empty list if agency is None.")
        with CatchAndAssertWarnings(
                clear=['obspy.io.stationxml.core'],
                expected=[(ObsPyDeprecationWarning, regex)]):
            assert station.operators[0].agencies[0] == "Agency 1"
        assert station.operators[0].contacts[0].names[0] == "This person"
        assert station.operators[0].contacts[0].names[1] == \
            "has multiple names!"
        assert len(station.operators[0].contacts[0].agencies) == 3
        assert station.operators[0].contacts[0].agencies[0] == "And also"
        assert station.operators[0].contacts[0].agencies[1] == "many"
        assert station.operators[0].contacts[0].agencies[2] == "many Agencies"
        assert len(station.operators[0].contacts[0].emails) == 4
        assert station.operators[0].contacts[0].emails[0] == "email1@mail.com"
        assert station.operators[0].contacts[0].emails[1] == "email2@mail.com"
        assert station.operators[0].contacts[0].emails[2] == "email3@mail.com"
        assert station.operators[0].contacts[0].emails[3] == "email4@mail.com"
        assert len(station.operators[0].contacts[0].phones) == 2
        assert station.operators[0].contacts[0].phones[0].description == \
            "phone number 1"
        assert station.operators[0].contacts[0].phones[0].country_code == 49
        assert station.operators[0].contacts[0].phones[0].area_code == 123
        assert station.operators[0].contacts[0].phones[0].phone_number == \
            "456-7890"
        assert station.operators[0].contacts[0].phones[1].description == \
            "phone number 2"
        assert station.operators[0].contacts[0].phones[1].country_code == 34
        assert station.operators[0].contacts[0].phones[1].area_code == 321
        assert station.operators[0].contacts[0].phones[1].phone_number == \
            "129-7890"
        assert station.operators[0].contacts[1].names[0] == "Name"
        assert station.operators[0].contacts[1].agencies[0] == "Agency"
        assert station.operators[0].contacts[1].emails[0] == "email@mail.com"
        assert station.operators[0].contacts[1].phones[0].description == \
            "phone number 1"
        assert station.operators[0].contacts[1].phones[0].country_code == 49
        assert station.operators[0].contacts[1].phones[0].area_code == 123
        assert station.operators[0].contacts[1].phones[0].phone_number == \
            "456-7890"
        assert station.operators[0].website == "http://www.web.site"

        assert station.operators[1].agency == "Agency"
        assert station.operators[1].contacts[0].names[0] == "New Name"
        assert station.operators[1].contacts[0].agencies[0] == "Agency"
        assert station.operators[1].contacts[0].emails[0] == "email@mail.com"
        assert station.operators[1].contacts[0].phones[0].description == \
            "phone number 1"
        assert station.operators[1].contacts[0].phones[0].country_code == 49
        assert station.operators[1].contacts[0].phones[0].area_code == 123
        assert station.operators[1].contacts[0].phones[0].phone_number == \
            "456-7890"
        assert station.operators[1].website == "http://www.web.site"

        assert station.creation_date == obspy.UTCDateTime(1990, 5, 5)
        assert station.termination_date == obspy.UTCDateTime(2009, 5, 5)
        assert station.total_number_of_channels == 100
        assert station.selected_number_of_channels == 1

        assert len(station.external_references) == 2
        assert station.external_references[0].uri == \
            "http://path.to/something"
        assert station.external_references[0].description == \
            "Some description"
        assert station.external_references[1].uri == \
            "http://path.to/something/else"
        assert station.external_references[1].description == \
            "Some other description"

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

    def test_reading_and_writing_channel_with_response(self, testdata):
        """
        Test the reading and writing of a single channel including a
        multi-stage response object.
        """
        filename = testdata["IRIS_single_channel_with_response.xml"]
        inv = obspy.read_inventory(filename)
        assert inv.source == "IRIS-DMC"
        assert inv.sender == "IRIS-DMC"
        assert inv.created == obspy.UTCDateTime("2013-04-16T06:15:28")
        # Assert that precisely one channel object has been created.
        assert len(inv.networks) == 1
        assert len(inv.networks[0].stations) == 1
        assert len(inv.networks[0].stations[0].channels) == 1
        network = inv.networks[0]
        station = network.stations[0]
        channel = station.channels[0]
        # Assert some fields of the network. This is extensively tested
        # elsewhere.
        assert network.code == "IU"
        assert network.start_date == obspy.UTCDateTime("1988-01-01T00:00:00")
        assert network.end_date == obspy.UTCDateTime("2500-12-12T23:59:59")
        assert network.description == \
            "Global Seismograph Network (GSN - IRIS/USGS)"
        # Assert a few fields of the station. This is extensively tested
        # elsewhere.
        assert station.code == "ANMO"
        assert station.latitude == 34.94591
        assert station.longitude == -106.4572
        assert station.elevation == 1820.0
        assert station.site.name == "Albuquerque, New Mexico, USA"
        # Start to assert the channel reading.
        assert channel.code == "BHZ"
        assert channel.location_code == "10"
        assert channel.start_date == obspy.UTCDateTime("2012-03-13T08:10:00")
        assert channel.end_date == obspy.UTCDateTime("2599-12-31T23:59:59")
        assert channel.restricted_status == "open"
        assert channel.latitude == 34.945913
        assert channel.longitude == -106.457122
        assert channel.elevation == 1759.0
        assert channel.depth == 57.0
        assert channel.azimuth == 0.0
        assert channel.dip == -90.0
        assert channel.types == ["CONTINUOUS", "GEOPHYSICAL"]
        assert channel.sample_rate == 40.0
        assert channel.clock_drift_in_seconds_per_sample == 0.0
        assert channel.sensor.type == \
            "Guralp CMG3-T Seismometer (borehole)"
        # Check the response.
        response = channel.response
        sensitivity = response.instrument_sensitivity
        assert sensitivity.value == 3.31283E10
        assert sensitivity.frequency == 0.02
        assert sensitivity.input_units == "M/S"
        assert sensitivity.input_units_description == \
            "Velocity in Meters Per Second"
        assert sensitivity.output_units == "COUNTS"
        assert sensitivity.output_units_description == "Digital Counts"
        # Assert that there are three stages.
        assert len(response.response_stages) == 3

    def test_stationxml_with_availability(self, testdata):
        """
        A variant of StationXML has support for availability information.
        Make sure this works.
        """
        filename = testdata["stationxml_with_availability.xml"]
        inv = obspy.read_inventory(filename, format="stationxml")
        channel = inv[0][0][0]
        assert channel.data_availability.start == \
            obspy.UTCDateTime("1998-10-26T20:35:58")
        assert channel.data_availability.end == \
            obspy.UTCDateTime("2014-07-21T12:00:00")

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

    def test_parse_file_with_no_default_namespace(self, testdata):
        """
        Tests that reading a file with no default namespace works fine.

        See #1060.
        """
        filename = testdata["no_default_namespace.xml"]
        inv = obspy.read_inventory(filename)
        # Very small file with almost no content.
        assert len(inv.networks) == 1
        assert inv[0].code == "XX"

    def test_parse_file_with_schema_2(self, testdata):
        """
        Reading a StationXML file version 2.0
        """
        filename = testdata["version20.xml"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            inv = obspy.read_inventory(filename)
        assert len(w) == 1
        assert 'StationXML file has version 2.0' in str(w[0].message)

        # Very small file with almost no content.
        assert len(inv.networks) == 1
        assert inv[0].code == "XX"

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
        data = re.sub(r'\s+', ' ', data)

        assert '<Zero number="0"> <Real>0.0</Real> ' \
            '<Imaginary>1.0</Imaginary> </Zero>' in data
        assert '<Zero number="1"> <Real>2.0</Real> ' \
            '<Imaginary>3.0</Imaginary> </Zero>' in data
        assert '<Pole number="0"> <Real>0.0</Real> ' \
            '<Imaginary>1.0</Imaginary> </Pole>' in data
        assert '<Pole number="1"> <Real>2.0</Real> ' \
            '<Imaginary>3.0</Imaginary> </Pole>' in data

    def test_write_with_extra_tags_namespace_redef(self, testdata):
        """
        Tests the exceptions are raised when namespaces
        are redefined.
        """
        filename = testdata["stationxml_with_availability.xml"]
        # read the StationXML with availability
        inv = obspy.read_inventory(filename)
        with NamedTemporaryFile() as tf:
            # manually add custom namespace definition
            tmpfile = tf.name
            # assert that namespace prefix of None raises ValueError
            mynsmap = {None: 'http://bad.custom.ns/'}
            with pytest.raises(ValueError):
                inv.write(path_or_file_object=tmpfile,
                          format="STATIONXML", nsmap=mynsmap)

    def test_write_with_extra_tags_without_read_extra(self):
        """
        Tests that a Inventory object that was instantiated with
        custom namespace tags and attributes is written correctly.
        """
        # read the default inventory
        inv = obspy.read_inventory()
        # manually add extra to the dictionary
        network = inv[0]
        network.extra = {}
        ns = 'http://test.myns.ns/'
        # manually add a new custom namespace tag and attribute to the
        # inventory
        network.extra['mynsNetworkTag'] = AttribDict({
            'value': 'mynsNetworkTagValue',
            'namespace': ns})
        network.extra['mynsNetworkAttrib'] = AttribDict({
            'value': 'mynsNetworkAttribValue',
            'namespace': ns,
            'type': 'attribute'})
        station = inv[0][0]
        station.extra = {}
        station.extra['mynsStationTag'] = AttribDict({
            'value': 'mynsStationTagValue',
            'namespace': ns})
        station.extra['mynsStationAttrib'] = AttribDict({
            'value': 'mynsStationAttribValue',
            'namespace': ns,
            'type': 'attribute'})
        channel = inv[0][0][0]
        # add data availability to inventory
        channel.data_availability = DataAvailability(
            start=UTCDateTime('1998-10-26T20:35:58+00:00'),
            end=UTCDateTime('2014-07-21T12:00:00+00:00'))
        channel.extra = {}
        channel.extra['mynsChannelTag'] = AttribDict({
            'value': 'mynsChannelTagValue', 'namespace': ns})
        channel.extra['mynsChannelAttrib'] = AttribDict({
            'value': 'mynsChannelAttribValue',
            'namespace': ns,
            'type': 'attribute'})
        # add nested tags
        nested_tag = AttribDict()
        nested_tag.namespace = ns
        nested_tag.value = AttribDict()
        # add two nested tags
        nested_tag.value.my_nested_tag1 = AttribDict()
        nested_tag.value.my_nested_tag1.namespace = ns
        nested_tag.value.my_nested_tag1.value = 1.23E+10
        nested_tag.value.my_nested_tag2 = AttribDict()
        nested_tag.value.my_nested_tag2.namespace = ns
        nested_tag.value.my_nested_tag2.value = True
        nested_tag.value.my_nested_tag2.attrib = {'{%s}%s' % (
            ns, 'nestedAttribute1'): 'nestedAttributeValue1'}
        channel.extra['nested'] = nested_tag
        with NamedTemporaryFile() as tf:
            # manually add custom namespace definition
            tmpfile = tf.name
            # set namespace map to include only valid custom namespaces
            mynsmap = {'myns': ns}
            # write file with manually defined namespace map
            inv.write(tmpfile, format="STATIONXML", nsmap=mynsmap)
            # check contents
            with open(tmpfile, "rb") as fh:
                # enforce reproducible attribute orders through write_c14n
                obj = etree.fromstring(fh.read()).getroottree()
                buf = io.BytesIO()
                obj.write_c14n(buf)
                buf.seek(0, 0)
                content = buf.read()
            # check namespace definitions in root element
            expected = [
                b'xmlns="http://www.fdsn.org/xml/station/1"',
                b'xmlns:myns="http://test.myns.ns/"']
            for line in expected:
                assert line in content
            # check additional tags
            expected = [
                b'<myns:mynsNetworkTag>' +
                b'mynsNetworkTagValue' +
                b'</myns:mynsNetworkTag>',
                b'myns:mynsNetworkAttrib="mynsNetworkAttribValue"',
                b'<myns:mynsStationTag>' +
                b'mynsStationTagValue' +
                b'</myns:mynsStationTag>',
                b'myns:mynsStationAttrib="mynsStationAttribValue"',
                b'<myns:mynsChannelTag>' +
                b'mynsChannelTagValue' +
                b'</myns:mynsChannelTag>',
                b'myns:mynsChannelAttrib="mynsChannelAttribValue"',
                b'<myns:nested>',
                b'<myns:my_nested_tag1>' +
                b'12300000000.0' +
                b'</myns:my_nested_tag1>',
                b'<myns:my_nested_tag2 ' +
                b'myns:nestedAttribute1="nestedAttributeValue1">' +
                b'True' +
                b'</myns:my_nested_tag2>',
                b'</myns:nested>'
            ]
            for line in expected:
                assert line in content

    def test_write_with_extra_tags_and_read(self, testdata):
        """
        First tests that a StationXML file with additional
        custom "extra" tags gets written correctly. Then
        tests that when reading the written file again the
        extra tags are parsed correctly.
        """
        filename = testdata[
            "IRIS_single_channel_with_response_custom_tags.xml"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inv = obspy.read_inventory(filename)
            assert len(w) == 0
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # write file
            inv.write(tmpfile, format="STATIONXML")
            # check contents
            with open(tmpfile, "rb") as fh:
                # enforce reproducible attribute orders through write_c14n
                obj = etree.fromstring(fh.read()).getroottree()
                buf = io.BytesIO()
                obj.write_c14n(buf)
                buf.seek(0, 0)
                content = buf.read()
            # check namespace definitions in root element
            expected = [b'xmlns="http://www.fdsn.org/xml/station/1"',
                        b'xmlns:test="http://just.a.test/xmlns/1"'
                        ]
            for line in expected:
                assert line in content
            # check custom tags, nested custom tags, and attributes
            # at every level of the StationXML hierarchy
            expected = [
                # root
                b'test:customRootAttrib="testRootAttribute"',
                b'<test:CustomRootTag>testRootTag</test:CustomRootTag>',
                b'<test:CustomNestedRootTag>',
                b'<test:NestedTag1 nestedTagAttrib="testNestedAttribute">' +
                b'nestedRootTag1' +
                b'</test:NestedTag1>',
                b'<test:NestedTag2>nestedRootTag2</test:NestedTag2>',
                b'</test:CustomNestedRootTag>',
                # network
                b'test:customNetworkAttrib="testNetworkAttribute"',
                b'<test:CustomNetworkTag>' +
                b'testNetworkTag' +
                b'</test:CustomNetworkTag>',
                b'<test:CustomNestedNetworkTag>',
                b'<test:NestedTag1>nestedNetworkTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedNetworkTag2</test:NestedTag2>',
                b'</test:CustomNestedNetworkTag>',
                # station
                b'test:customStationAttrib="testStationAttribute"',
                b'<test:CustomStationTag>' +
                b'testStationTag' +
                b'</test:CustomStationTag>',
                b'<test:CustomNestedStationTag>',
                b'<test:NestedTag1>nestedStationTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedStationTag2</test:NestedTag2>',
                b'</test:CustomNestedStationTag>',
                # comment
                b'test:customCommentAttrib="testCommentAttribute"',
                b'<test:CustomCommentTag>' +
                b'testCommentTag' +
                b'</test:CustomCommentTag>',
                b'<test:CustomNestedCommentTag>',
                b'<test:NestedTag1>nestedCommentTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedCommentTag2</test:NestedTag2>',
                b'</test:CustomNestedCommentTag>',
                # person
                b'test:customPersonAttrib="testPersonAttribute"',
                b'<test:CustomPersonTag>testPersonTag</test:CustomPersonTag>',
                b'<test:CustomNestedPersonTag>',
                b'<test:NestedTag1>nestedPersonTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedPersonTag2</test:NestedTag2>',
                b'</test:CustomNestedPersonTag>',
                # phone
                b'test:customPhoneAttrib="testPhoneAttribute"',
                b'<test:CustomPhoneTag>testPhoneTag</test:CustomPhoneTag>',
                b'<test:CustomNestedPhoneTag>',
                b'<test:NestedTag1>nestedPhoneTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedPhoneTag2</test:NestedTag2>',
                b'</test:CustomNestedPhoneTag>',
                # site
                b'test:customSiteAttrib="testSiteAttribute"',
                b'<test:CustomSiteTag>testSiteTag</test:CustomSiteTag>',
                b'<test:CustomNestedSiteTag>',
                b'<test:NestedTag1>nestedSiteTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedSiteTag2</test:NestedTag2>',
                b'</test:CustomNestedSiteTag>',
                # equipment
                b'test:customEquipmentAttrib="testEquipmentAttribute"',
                b'<test:CustomEquipmentTag>' +
                b'testEquipmentTag' +
                b'</test:CustomEquipmentTag>',
                b'<test:CustomNestedEquipmentTag>',
                b'<test:NestedTag1>nestedEquipmentTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedEquipmentTag2</test:NestedTag2>',
                b'</test:CustomNestedEquipmentTag>',
                # operator
                b'test:customOperatorAttrib="testOperatorAttribute"',
                b'<test:CustomOperatorTag>' +
                b'testOperatorTag' +
                b'</test:CustomOperatorTag>',
                b'<test:CustomNestedOperatorTag>',
                b'<test:NestedTag1>nestedOperatorTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedOperatorTag2</test:NestedTag2>',
                b'</test:CustomNestedOperatorTag>',
                # external reference
                b'test:customExtRefAttrib="testExtRefAttribute"',
                b'<test:CustomExtRefTag>testExtRefTag</test:CustomExtRefTag>',
                b'<test:CustomNestedExtRefTag>',
                b'<test:NestedTag1>nestedExtRefTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedExtRefTag2</test:NestedTag2>',
                b'</test:CustomNestedExtRefTag>',
                # channel
                b'test:customChannelAttrib="testChannelAttribute"',
                b'<test:CustomChannelTag>' +
                b'testChannelTag' +
                b'</test:CustomChannelTag>',
                b'<test:CustomNestedChannelTag>',
                b'<test:NestedTag1>nestedChannelTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedChannelTag2</test:NestedTag2>',
                b'</test:CustomNestedChannelTag>',
                # response
                b'test:customResponseAttrib="testResponseAttribute"',
                b'<test:CustomResponseTag>' +
                b'testResponseTag' +
                b'</test:CustomResponseTag>',
                b'<test:CustomNestedResponseTag>',
                b'<test:NestedTag1>nestedResponseTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedResponseTag2</test:NestedTag2>',
                b'</test:CustomNestedResponseTag>',
                # data availability
                b'test:customDAAttrib="testDAAttribute"',
                b'<test:CustomDATag>testDATag</test:CustomDATag>',
                b'<test:CustomNestedDATag>',
                b'<test:NestedTag1>nestedDATag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedDATag2</test:NestedTag2>',
                b'</test:CustomNestedDATag>',
                # response stage (PolesZeros response stage)
                b'test:customStagePZAttrib="testStagePZAttribute"',
                b'<test:CustomStagePZTag>' +
                b'testStagePZTag' +
                b'</test:CustomStagePZTag>',
                b'<test:CustomNestedStagePZTag>',
                b'<test:NestedTag1>nestedStagePZTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedStagePZTag2</test:NestedTag2>',
                b'</test:CustomNestedStagePZTag>',
                # response stage (Coefficients response stage)
                b'test:customStageCoefAttrib="testStageCoefAttribute"',
                b'<test:CustomStageCoefTag>' +
                b'testStageCoefTag' +
                b'</test:CustomStageCoefTag>',
                b'<test:CustomNestedStageCoefTag>',
                b'<test:NestedTag1>nestedStageCoefTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedStageCoefTag2</test:NestedTag2>',
                b'</test:CustomNestedStageCoefTag>',
                # instrument sensitivity
                b'test:customSensitivityAttrib="testSensitivityAttribute"',
                b'<test:CustomSensitivityTag>' +
                b'testSensitivityTag' +
                b'</test:CustomSensitivityTag>',
                b'<test:CustomNestedSensitivityTag>',
                b'<test:NestedTag1>nestedSensitivityTag1</test:NestedTag1>',
                b'<test:NestedTag2>nestedSensitivityTag2</test:NestedTag2>',
                b'</test:CustomNestedSensitivityTag>'
            ]
            for line in expected:
                assert line in content
            # now, read again to test if it's parsed correctly..
            inv = obspy.read_inventory(tmpfile)

    def test_reading_file_with_empty_channel_object(self, testdata):
        """
        Tests reading a file with an empty channel object. This is strictly
        speaking not valid but we are forgiving.
        """
        filename = testdata["empty_channel.xml"]
        inv = obspy.read_inventory(filename)
        assert inv.get_contents() == \
            {'networks': ['IV'], 'stations': ['IV.LATE (Latera)'],
             'channels': []}

    def test_reading_channel_without_coordinates(self, testdata):
        """
        Tests reading a file with an empty channel object. This is strictly
        speaking not valid but we are forgiving.
        """
        filename = testdata["channel_without_coordinates.xml"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inv = obspy.read_inventory(filename)

        # Should raise a warning that it could not read the channel without
        # coordinates.
        assert len(w) == 1
        assert w[0].message.args[0] == \
            ("Channel 00.BHZ of station LATE does not have a complete set of "
             "coordinates (latitude, longitude), elevation and depth and thus "
             "it cannot be read. It will not be part of "
             "the final inventory object.")

        assert inv.get_contents() == \
            {'networks': ['IV'], 'stations': ['IV.LATE (Latera)'],
             'channels': []}

    def test_units_during_identity_stage(self):
        """
        """
        t = obspy.UTCDateTime(2017, 1, 1)
        inv = obspy.read_inventory().select(station="RJOB", channel="EHZ",
                                            time=t)
        response = inv.get_response("BW.RJOB..EHZ", t)
        response.response_stages[0].input_units_description = "M/S"
        response.response_stages[0].output_units_description = "Volts"
        rstage_2 = ResponseStage(2, 1, 1, "V", "V",
                                 input_units_description="Volts",
                                 output_units_description="Volts")
        rstage_3 = ResponseStage(3, 1, 1, "V", "V",
                                 input_units_description="Volts",
                                 output_units_description="Volts")
        response.response_stages.insert(1, rstage_2)
        response.response_stages.insert(2, rstage_3)
        for i, rstage in enumerate(response.response_stages[3:]):
            rstage.stage_sequence_number = i + 4

        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml", validate=True)
            buf.seek(0, 0)
            inv_2 = obspy.read_inventory(buf)

        response_2 = inv_2.get_response("BW.RJOB..EHZ", t)

        assert response == response_2
        assert response_2.response_stages[1].input_units == "V"
        assert response_2.response_stages[1].output_units == "V"
        assert response_2.response_stages[1].input_units_description == "Volts"
        assert response_2.response_stages[1].output_units_description == \
            "Volts"
        assert response_2.response_stages[2].input_units == "V"
        assert response_2.response_stages[2].output_units == "V"
        assert response_2.response_stages[2].input_units_description == "Volts"
        assert response_2.response_stages[2].output_units_description == \
            "Volts"

        # Also try from the other side.
        inv = obspy.read_inventory().select(station="RJOB", channel="EHZ",
                                            time=t)
        response = inv.get_response("BW.RJOB..EHZ", t)
        response.response_stages[0].input_units = None
        response.response_stages[0].output_units = None
        response.response_stages[0].input_units_description = None
        response.response_stages[0].output_units_description = None
        rstage_2 = ResponseStage(2, 1, 1, "V", "V",
                                 input_units_description="Volts",
                                 output_units_description="Volts")
        rstage_3 = ResponseStage(3, 1, 1, "V", "V",
                                 input_units_description="Volts",
                                 output_units_description="Volts")
        response.response_stages.insert(1, rstage_2)
        response.response_stages.insert(2, rstage_3)
        for i, rstage in enumerate(response.response_stages[3:]):
            rstage.stage_sequence_number = i + 4

        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml")
            buf.seek(0, 0)
            inv_2 = obspy.read_inventory(buf)

        response_2 = inv_2.get_response("BW.RJOB..EHZ", t)
        # Set these to None manually as the autocorrection during parsing will
        # set it.
        response_2.response_stages[0].input_units = None
        response_2.response_stages[0].input_units_description = None

        assert response == response_2
        assert response_2.response_stages[1].input_units == "V"
        assert response_2.response_stages[1].output_units == "V"
        assert response_2.response_stages[1].input_units_description == "Volts"
        assert response_2.response_stages[1].output_units_description == \
            "Volts"
        assert response_2.response_stages[2].input_units == "V"
        assert response_2.response_stages[2].output_units == "V"
        assert response_2.response_stages[2].input_units_description == "Volts"
        assert response_2.response_stages[2].output_units_description == \
            "Volts"

    def test_reading_full_stationxml_1_0_file(self, testdata):
        """
        Tests reading a fully filled StationXML 1.0 file.
        """
        filename = testdata["full_random_stationxml_1_0.xml"]
        inv = obspy.read_inventory(filename, format='STATIONXML')
        lats = [cha.latitude for net in inv for sta in net for cha in sta]
        # for now just check that all expected channels are there.. test could
        # be much improved
        assert lats == [
            -53.12, 44.77, 63.39, 12.46, -13.16, -84.44, 43.9, -88.41]

    def test_read_with_level(self, testdata):
        """
        Tests reading StationXML with specifying the level of detail.
        """
        path = testdata['stationxml_BK.CMB.__.LKS.xml']
        inv_stationxml_no_level = _read_stationxml(path)
        inv_stationxml_response = _read_stationxml(path, level='response')
        inv_stationxml_channel = _read_stationxml(path, level='channel')
        inv_stationxml_station = _read_stationxml(path, level='station')
        inv_stationxml_network = _read_stationxml(path, level='network')
        # test reading through plugin
        assert obspy.read_inventory(
            path, format='STATIONXML', level='station') == \
            inv_stationxml_station
        # test reading default which should be equivalent to reading response
        # level
        assert inv_stationxml_no_level == inv_stationxml_response
        # test reading response level
        assert len(inv_stationxml_response) == 1
        assert len(inv_stationxml_response[0]) == 1
        assert len(inv_stationxml_response[0][0]) == 1
        assert inv_stationxml_response[0][0][0].response is not None
        # test reading channel level
        assert len(inv_stationxml_channel) == 1
        assert len(inv_stationxml_channel[0]) == 1
        assert len(inv_stationxml_channel[0][0]) == 1
        assert inv_stationxml_channel[0][0][0].response is None
        # test reading station level
        assert len(inv_stationxml_station) == 1
        assert len(inv_stationxml_station[0]) == 1
        assert len(inv_stationxml_station[0][0]) == 0
        # test reading station level
        assert len(inv_stationxml_network) == 1
        assert len(inv_stationxml_network[0]) == 0

    def test_read_basic_responsestage_with_decimation(self, testdata):
        """
        Make sure basic ResponseStage elements that have decimation information
        do not lose that information.
        """
        path = testdata['F1_423_small.xml']
        inv = _read_stationxml(path)
        stage = inv[0][0][0].response.response_stages[0]
        assert stage.decimation_correction == 0.4
        assert stage.decimation_delay == 0.5
        assert stage.decimation_factor == 2
        assert stage.decimation_input_sample_rate == 1024000.0
        assert stage.decimation_offset == 1
