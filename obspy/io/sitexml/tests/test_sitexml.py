#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the SiteXML reader and writer.

:author:
    Kiriaki Konstantinidou (kiriaki@itsak.gr), 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
import re
import warnings

from lxml import etree
import obspy
import pytest
from obspy.io.sitexml.core import (Analysis, LiteratureSource, SERASiteOwner,
                                   SiteDescription, ValueWithUncertainty,
                                   VelocityProfile, VelocityProfileData)
from obspy.io.sitexml.util import SiteXMLValidationError
from obspy.io.sitexml.sitexml import (_is_sitexml, _read_site_description,
                                      _read_site_owner, read_sitexml)
from obspy.io.sitexml.sitexml import write_sitexml

class TestSiteXML():
    """
    """
    def _assert_site_xml_equality(self, xml_file_buffer,
                                     expected_xml_file_buffer):
        """
        Helper function comparing two BytesIO buffers contain SiteXML
        files.
        """
        new_lines = [_i.decode('utf-8').strip().replace("'", '"')
                     for _i in xml_file_buffer.read().splitlines()]
        org_lines = [_i.decode('utf-8').strip().replace("'", '"')
                     for _i in expected_xml_file_buffer.read().splitlines()]

        # Remove the module lines from the original file.
        #org_lines = [_i.strip() for _i in org_lines
        #             if not _i.strip().startswith("<!--")]

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

    def _write_and_compare(self, orig_filename, sera_site):
        """
        Helper function for creating two BytesIO buffers contain SiteXML
        files to be compared by _assert_site_xml_equality().
        
        :type orig_filename: str
        :param orig_filename: Name of the file to read from siteXML doc
        :type sera_site: :class:`~obspy.core.io.sitexml.core.SERASite` 
        :param sera_site: A SERASite object containing site metadata from orig_filename
        """
        # Read orig_filename into orig_xml_file_buffer BytesIO buffer
        #
        with open(orig_filename, "rb") as open_file:
            orig_xml_file_buffer = io.BytesIO(open_file.read())
        orig_xml_file_buffer.seek(0, 0)

        # Write sera_site into new_xml_file_buffer BytesIO buffer
        #
        new_xml_file_buffer = io.BytesIO()
        write_sitexml(sera_site, new_xml_file_buffer, validate=True)
        new_xml_file_buffer.seek(0, 0)
        
        # Compare the two buffers
        #
        self._assert_site_xml_equality(
            new_xml_file_buffer, orig_xml_file_buffer)
        
    def test_is_sitexml(self, testdata, datapath):
        """
        Tests the _is_sitexml() function.
        """
        # Check positives.
        sitexmls = [testdata["full_sitexml.xml"]]
        for stat in sitexmls:
            assert _is_sitexml(stat)
        
        # Check some negatives.
        #not_sitexmls = [
        #    "wrong_sitexml.xml", "input_csv/site_description.csv",
        #    "input_excel/sera_site_all.xlsx"]
        #not_sitexmls = [datapath / name
        #                   for name in not_sitexmls]
        #for stat in not_sitexmls:
        #    assert not _is_sitexml(stat)
    
    def test_read_and_write_minimal_file(self, testdata):
        """
        Test that reading and writing of a minimal SiteXML document, 
        with the least possible tags, works.
        """
        filename = testdata["minimal_sitexml.xml"]
        sera_site = read_sitexml(filename)

        # Write it again. Also validate it to get more confidence.
        self._write_and_compare(filename, sera_site)

    def test_read_sitexml_accepts_seekable_file_like_objects(self, testdata):
        filename = testdata["minimal_sitexml.xml"]

        with open(filename, "rb") as fh:
            xml_buffer = io.BytesIO(fh.read())

        sera_site = read_sitexml(xml_buffer)

        assert sera_site is not None
        assert sera_site.site_owner is not None
        assert sera_site.site_description is not None

    def test_read_sitexml_raises_sitexml_validation_error_for_invalid_xml(self):
        xml_buffer = io.BytesIO(b"<not-sitexml />")

        with pytest.raises(SiteXMLValidationError):
            read_sitexml(xml_buffer)

    def test_analysis_requires_schema_required_ids(self):
        with pytest.raises(SiteXMLValidationError):
            Analysis(resource_id=None,
                     site_descriptionID="quakeml:domain.ab/site_description/001")

        with pytest.raises(SiteXMLValidationError):
            Analysis(resource_id="quakeml:domain.ab/analysis/001",
                     site_descriptionID=None)

        with pytest.raises(SiteXMLValidationError):
            Analysis(resource_id="",
                     site_descriptionID="quakeml:domain.ab/site_description/001")

    def test_site_description_requires_schema_required_fields(self):
        with pytest.raises(SiteXMLValidationError):
            SiteDescription(resource_id=None, latitude=45.0, longitude=7.0)

        with pytest.raises(SiteXMLValidationError):
            SiteDescription(
                resource_id="quakeml:domain.ab/site_description/001",
                latitude=None, longitude=7.0)

        with pytest.raises(SiteXMLValidationError):
            SiteDescription(
                resource_id="quakeml:domain.ab/site_description/001",
                latitude=45.0, longitude=None)

    def test_literature_source_requires_schema_required_fields(self):
        with pytest.raises(SiteXMLValidationError):
            LiteratureSource(title=None, first_author="Author A.")

        with pytest.raises(SiteXMLValidationError):
            LiteratureSource(title="Some title", first_author=None)

        with pytest.raises(SiteXMLValidationError):
            LiteratureSource(title=" ", first_author="Author A.")

    def test_literature_source_year_uses_schema_string_type(self):
        literature_source = LiteratureSource(
            title="Some title", first_author="Author A.", year=2018)

        assert literature_source.year == "2018"

        with pytest.raises(SiteXMLValidationError):
            LiteratureSource(
                title="Some title", first_author="Author A.", year="18")

    def test_site_owner_requires_schema_required_fields(self):
        with pytest.raises(SiteXMLValidationError):
            SERASiteOwner(
                owner_codename=None,
                owner_fullname="Test Owner",
                person_firstname="Name",
                person_lastname="Surname",
                person_mbox="someemail@domain.ab")

        with pytest.raises(SiteXMLValidationError):
            SERASiteOwner(
                owner_codename="TEST",
                owner_fullname="Test Owner",
                person_firstname="Name",
                person_lastname=None,
                person_mbox="someemail@domain.ab")

    def test_read_site_description_requires_schema_required_fields(self):
        element = etree.fromstring(
            b"""
            <siteDescription xmlns="http://www.orfeus-eu.org/xml/site/1">
                <latitude>45.0</latitude>
                <longitude>7.0</longitude>
            </siteDescription>
            """)

        with pytest.raises(SiteXMLValidationError):
            _read_site_description(element)

    def test_read_site_owner_requires_schema_required_contact_person(self):
        element = etree.fromstring(
            b"""
            <siteOwner xmlns="http://www.orfeus-eu.org/xml/site/1">
                <codeName>TEST</codeName>
                <fullName>Test Owner</fullName>
            </siteOwner>
            """)

        with pytest.raises(SiteXMLValidationError):
            _read_site_owner(element)

    @pytest.mark.parametrize("topography_a, topography_b", [
        ("T1", None),
        (None, "Flat"),
    ])
    def test_write_site_topography_requires_at_least_one_schema(
            self, testdata, topography_a, topography_b):
        filename = testdata["minimal_sitexml.xml"]
        sera_site = read_sitexml(filename)
        sera_site.site_description.topographyA = topography_a
        sera_site.site_description.topographyB = topography_b

        xml_buffer = io.BytesIO()
        write_sitexml(sera_site, xml_buffer, validate=True)
        
    def test_read_and_write_full_file(self, testdata):
        """
        Test that reading and writing of a full SiteXML document with all
        possible tags works.
        """
        filename = testdata["full_sitexml.xml"]
        sera_site = read_sitexml(filename)

        # Write it again. Also validate it to get more confidence.
        self._write_and_compare(filename, sera_site)
        
    def test_reading_and_writing_full_siteowner_tag(self, testdata):
        """
        Tests reading and writing a full SiteXML <siteOwner> tag.
        """
        filename = testdata["full_siteowner.xml"]
        sera_site = read_sitexml(filename)

        assert sera_site.site_owner.owner_codename == "SITEOWNER"
        assert sera_site.site_owner.owner_fullname == "Site Owner Full Name"
        assert sera_site.site_owner.ownerID == "quakeml:domain.ab/siteOwner/001"

        assert sera_site.site_owner.personID == "quakeml:domain.ab/person/001"
        assert sera_site.site_owner.person_firstname == "Name"
        assert sera_site.site_owner.person_lastname == "Surname"
        assert sera_site.site_owner.person_mbox == "someemail@domain.ab"
        assert sera_site.site_owner.person_homepage == "https://www.domain.ab/person"

        assert sera_site.site_owner.institutionID == "quakeml:domain.ab/institution/001"
        assert sera_site.site_owner.institution_name == "INSTITUTION_ABBR"
        assert sera_site.site_owner.institution_mbox == "info@domain.ab"
        assert sera_site.site_owner.institution_phone == "+30 123 456789"
        assert sera_site.site_owner.institution_homepage == "http://www.domain.ab"

        assert sera_site.site_owner.affiliation_department == "Seismology"
        assert sera_site.site_owner.affiliation_function == "Senior researcher"

        assert sera_site.site_owner.address_street == "Some streetAddress"
        assert sera_site.site_owner.address_locality == "City" 
        assert sera_site.site_owner.address_postal_code == "12345"
    
        assert sera_site.site_owner.address_country == "Somecountry" 
        assert sera_site.site_owner.address_country_code == "AB" 

        # Write it again and compare to the original file.
        self._write_and_compare(filename, sera_site)
        
    def test_reading_and_writing_full_sitedescription_tag(self, testdata):
        """
        Tests reading and writing a full SiteXML <siteDescription> tag.
        """
        filename = testdata["full_sitedescription.xml"]
        sera_site = read_sitexml(filename)

        assert sera_site.site_description is not None
        assert sera_site.site_description.resource_id == "quakeml:domain.ab/site_description/001"
        assert sera_site.site_description.station_code == "ABCD"
        assert sera_site.site_description.latitude == 45.137174
        assert sera_site.site_description.longitude == 5.998905

        assert sera_site.site_description.altitude == 239.0
        assert sera_site.site_description.min_distance_from_station == 10.3
        assert sera_site.site_description.max_distance_from_station == 10.3

        assert sera_site.site_description.topographyA == "T1"
        assert sera_site.site_description.topographyB == "Valley"
        assert sera_site.site_description.morphology == "Plain"

        assert sera_site.site_description.ec8 is not None
        assert sera_site.site_description.ec8.value == "C"
        assert sera_site.site_description.ec8.quality_index == 1.0

        assert sera_site.site_description.ec8.literature_source is not None
        ls = sera_site.site_description.ec8.literature_source
        assert ls.title == "Some title"
        assert ls.first_author == "Author A."
        assert ls.secondary_authors == "Author B., Author C."
        assert ls.year == "2018"
        assert ls.booktitle == "Some magazine"
        assert ls.doi == "10.1007/s10518-017-0135-5"
        assert ls.language == "en"

        assert sera_site.site_description.ec8.external_reference is not None
        external_ref = sera_site.site_description.ec8.external_reference
        assert external_ref.uri == "https://doi.org/10.1007/s10518-017-0135-5/"
        assert external_ref.description == "paper"

        assert sera_site.site_description.bedrock_depth is not None
        assert sera_site.site_description.bedrock_depth.value.value == 774.6218
        assert sera_site.site_description.bedrock_depth.value.uncertainty == 107.8669
        assert sera_site.site_description.bedrock_depth.quality_index == 0.5

        assert sera_site.site_description.h800 is not None
        assert sera_site.site_description.h800.value.value == 94.0736
        assert sera_site.site_description.h800.value.uncertainty == 15.5748
        assert sera_site.site_description.h800.quality_index == 0.43

        assert sera_site.site_description.geological_unit is not None
        assert sera_site.site_description.geological_unit.value == "Some geology"
        assert sera_site.site_description.geological_unit.quality_index == 0.25
        assert sera_site.site_description.geological_unit.geological_map_scale == "1:50000"
        assert sera_site.site_description.geological_unit.geological_unit_OGE == "Some description"
        
        # Write it again and compare to the original file.
        self._write_and_compare(filename, sera_site)

    def test_reading_and_writing_full_analysis_tag(self, testdata):
        """
        Tests reading and writing a full SiteXML <analysis> tag.
        """
        filename = testdata["full_analysis.xml"]
        sera_site = read_sitexml(filename)

        # Test that a preferred analysis ID is provided 
        assert sera_site.site_description.preferred_site_analysisID == \
            "quakeml:domain.ab/analysis/001"

        assert len(sera_site.analysis) == 1
        analysis = sera_site.analysis[0]
        assert analysis.resource_id == "quakeml:domain.ab/analysis/001"
        assert analysis.site_descriptionID == "quakeml:domain.ab/site_description/001"
        assert analysis.creation_date == obspy.UTCDateTime(2015, 11, 10)
        assert analysis.spt_logs_count == 2
        assert analysis.cpt_logs_count == 0
        assert analysis.borehole_logs_count == 0

        # Test Resonance Frequency specific tags
        assert analysis.resonance_frequency is not None
        f0 = analysis.resonance_frequency
        assert f0.value.value == 4.9962
        assert f0.value.uncertainty == 0.3494
        assert f0.quality_index == 0.8
        assert len(f0.methods) == 2
        assert f0.methods[0] == "HVSR EARTHQUAKE RECORDS"
        assert f0.methods[1] == "SSR EARTHQUAKE RECORDS"

        assert f0.literature_source is not None
        ls = f0.literature_source
        assert ls.title == "Some title"
        assert ls.first_author == "Author A."
        assert ls.secondary_authors == "Author B., Author C."
        assert ls.year == "2018"
        assert ls.booktitle == "Some magazine"
        assert ls.doi == "10.1007/s10518-017-0135-5"
        assert ls.language == "en"

        assert f0.external_reference is not None
        external_ref = f0.external_reference
        assert external_ref.uri == "https://doi.org/10.1007/s10518-017-0135-5/"
        assert external_ref.description == "paper"

        # Test VelocityS30 specific tags
        assert analysis.velocity_s30 is not None
        vs30 = analysis.velocity_s30
        assert vs30.value.value == 221.5954
        assert vs30.value.uncertainty == 18.34
        assert vs30.quality_index == 0.8
        assert len(vs30.methods) == 2
        assert vs30.methods[0] == "Geology"
        assert vs30.methods[1] == "Crosshole"
        assert vs30.method_combined_qindex == "1.2"
        assert vs30.manual_qindex == "1.0"

        # Write it again and compare to the original file.
        self._write_and_compare(filename, sera_site)

    def test_read_analysis_without_creation_time(self, testdata):
        """
        Tests reading a schema-valid <analysis> without optional creationTime.
        """
        filename = testdata["full_analysis.xml"]
        with open(filename, "rb") as fh:
            xml = fh.read()

        analysis_creation_time = (
            b"        <creationTime>2015-11-10T00:00:00.000000Z"
            b"</creationTime>\n"
        )
        xml = xml.replace(analysis_creation_time, b"", 1)

        sera_site = read_sitexml(io.BytesIO(xml))

        assert len(sera_site.analysis) == 1
        assert sera_site.analysis[0].creation_date is None

    def test_reading_and_writing_velocity_profile_tag(self, testdata):
        """
        Tests reading and writing a full SiteXML <velocityProfile> tag.
        """
        filename = testdata["full_analysis.xml"]
        sera_site = read_sitexml(filename)

        assert sera_site.analysis[0] is not None
        assert sera_site.analysis[0].velocity_profile_survey is not None

        vps = sera_site.analysis[0].velocity_profile_survey
        assert vps.quality_index == 1.0
        assert vps.velocity_profiles is not None
        assert len(vps.velocity_profiles) == 2

        # Test first/last layer from first velocity profile        
        vp = vps.velocity_profiles[0]
        assert vp.resource_id == "quakeml:domain.ab/velocity_profile/001"
        assert vp.layer_count == 8
        assert vp.velocity_profile_data is not None
        assert len(vp.velocity_profile_data) == 8
        
        vpd = vp.velocity_profile_data[0]               # First Layer
        assert vpd.velocityS.value == 118.08
        assert vpd.velocityS.uncertainty == 2.0
        assert vpd.top_depth.value == 0.0
        assert vpd.bottom_depth.value == 0.19
        
        vpd = vp.velocity_profile_data[7]               # Last Layer
        assert vpd.velocityS.value == 1108.37
        assert vpd.top_depth.value == 209.23

        # Test first/last layer from second velocity profile
        vp = vps.velocity_profiles[1]
        assert vp.resource_id == "quakeml:domain.ab/velocity_profile/002"
        assert vp.layer_count == 8
        assert vp.velocity_profile_data is not None
        assert len(vp.velocity_profile_data) == 8
        
        vpd = vp.velocity_profile_data[0]               # First Layer
        assert vpd.velocityS.value == 119.4
        assert vpd.top_depth.value == 0.0
        assert vpd.bottom_depth.value == 0.2
        
        vpd = vp.velocity_profile_data[7]               # Last Layer
        assert vpd.velocityS.value == 1097.0
        assert vpd.top_depth.value == 226.6

        # Write it again and compare to the original file.
        self._write_and_compare(filename, sera_site)

    def test_reading_velocity_profile_validates_layer_count(
            self, testdata, tmp_path):
        """
        Tests that layerCount matches the number of velocityProfileData items.
        """
        filename = testdata["full_analysis.xml"]
        invalid_xml = tmp_path / "invalid_layer_count.xml"
        invalid_xml.write_text(
            filename.read_text().replace(
                "<layerCount>8</layerCount>",
                "<layerCount>7</layerCount>",
                1),
            encoding="utf-8")

        with pytest.raises(SiteXMLValidationError, match="layer_count"):
            read_sitexml(invalid_xml)

    def test_reading_velocity_profile_without_layer_count(
            self, testdata, tmp_path):
        """
        Tests that missing layerCount is derived from velocityProfileData items.
        """
        filename = testdata["full_analysis.xml"]
        xml_text = filename.read_text(encoding="utf-8")
        invalid_xml = tmp_path / "missing_layer_count.xml"
        invalid_xml.write_text(
            xml_text.replace("<layerCount>8</layerCount>", "", 1),
            encoding="utf-8")

        sera_site = read_sitexml(invalid_xml)
        vp = sera_site.analysis[0].velocity_profile_survey.velocity_profiles[0]
        assert vp.layer_count == 8
        assert len(vp.velocity_profile_data) == 8

    def test_velocity_profile_requires_schema_required_fields(self):
        """
        Tests required VelocityProfile and VelocityProfileData fields.
        """
        top_depth = ValueWithUncertainty(0.0)
        layer = VelocityProfileData(top_depth=top_depth)

        with pytest.raises(SiteXMLValidationError, match="resource_id"):
            VelocityProfile(resource_id=None, velocity_profile_data=[layer])

        with pytest.raises(SiteXMLValidationError, match="velocity_profile_data"):
            VelocityProfile(
                resource_id="quakeml:domain.ab/velocity_profile/001",
                velocity_profile_data=None)

        with pytest.raises(SiteXMLValidationError, match="at least one layer"):
            VelocityProfile(
                resource_id="quakeml:domain.ab/velocity_profile/001",
                velocity_profile_data=[])

        with pytest.raises(SiteXMLValidationError, match="top_depth"):
            VelocityProfileData(top_depth=None)

    def test_velocity_profile_derives_layer_count_from_data(self):
        """
        Tests that layer_count is derived from velocity_profile_data when omitted.
        """
        layers = [
            VelocityProfileData(top_depth=ValueWithUncertainty(0.0)),
            VelocityProfileData(top_depth=ValueWithUncertainty(10.0)),
        ]
        profile = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/001",
            velocity_profile_data=layers)

        assert profile.layer_count == 2

    def test_reading_twice_raises_no_warning(self, testdata):
        """
        Tests that reading a siteXML file twice does not raise a warnings.
        """
        filename = testdata['full_analysis.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site1 = read_sitexml(filename)
            assert len(w) == 0
            site2 = read_sitexml(filename)
            assert len(w) == 0

        assert site1 == site2

    def test_deepcopy(self, testdata):
        """
        Tests that creating a deep copy of a siteXML object results in two identical objects.
        """
        filename = testdata['full_sitexml.xml']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site1 = read_sitexml(filename)
            assert len(w) == 0
            site2 = site1.copy()
            assert len(w) == 0

        assert site1 is not site2       # The two objects are not the same
        assert site1 == site2           # but they have the same data 

        # Write deep copied object site2 and compare to the original file.
        self._write_and_compare(filename, site2)
