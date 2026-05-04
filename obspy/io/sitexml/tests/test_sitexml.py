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
from pathlib import Path
import re
import warnings

from lxml import etree
import obspy
import pytest
from obspy.core.event import ResourceIdentifier
from obspy.core.inventory import Inventory, Network, Station
from obspy.core.inventory.util import ExternalReference, Operator, Person
from obspy.core.util.obspy_types import FloatWithUncertainties
from obspy.io.sitexml.core import (Analysis, EC8, LiteratureSource, Revision,
                                   SERASite, SERASiteOwner, SiteDescription,
                                   ResonanceFrequency,
                                   ValueWithUncertainty, VelocityProfile,
                                   VelocityProfileData,
                                   VelocityProfileSet, VelocityS30)
from obspy.io.sitexml import sitexml as sitexml_module
from obspy.io.sitexml.util import SiteXMLIOError, SiteXMLValidationError
from obspy.io.sitexml.quality_index import overall_quality_index
from obspy.io.sitexml.sitexml import (_is_sitexml, _read_site_description,
                                      _read_site_owner, add_sitexml_reference,
                                      read_sitexml,
                                      sitedict_to_sitexml,
                                      sitexml_to_sitedict)
from obspy.io.sitexml.sitexml import write_sitexml

class TestSiteXML():
    """
    """
    def _minimal_sera_site(self, station_code="XX.ABCD",
                           resource_id="quakeml:domain.ab/site/001"):
        """
        Build a minimal SERASite used by conversion/helper tests.
        """
        site_owner = SERASiteOwner(
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name",
            person_firstname="Name",
            person_lastname="Surname",
            person_mbox="someemail@domain.ab")
        site_description = SiteDescription(
            resource_id="quakeml:domain.ab/site_description/001",
            station_code=station_code,
            latitude=1.0,
            longitude=2.0)
        return SERASite(
            resource_id=resource_id,
            site_owner=site_owner,
            site_description=site_description)

    def test_get_sitexml_filename(self):
        """
        The default SiteXML filename follows station or site-ID identity.
        """
        station_site = self._minimal_sera_site(station_code="XX.ABCD")
        assert station_site.get_sitexml_filename() == "XX.ABCD.xml"
        assert station_site.get_sitexml_filename(
            obspy.UTCDateTime(2026, 1, 12, 3, 4, 5)) == (
                "Site_XX.ABCD_12-01-2026.xml")

        non_station_site = self._minimal_sera_site(station_code=None)
        assert non_station_site.get_sitexml_filename() == (
            "quakeml_domain_ab_site_001.xml")
        assert non_station_site.get_sitexml_filename(
            obspy.UTCDateTime(2026, 1, 12, 3, 4, 5)) == (
                "Site_quakeml_domain_ab_site_001_12-01-2026.xml")

    def test_sitedict_to_sitexml_uses_network_station_filename(
            self, tmp_path, monkeypatch):
        output_calls = []

        def fake_write_sitexml(sera_site, filename, validate):
            output_calls.append((sera_site, filename, validate))

        monkeypatch.setattr(
            sitexml_module, "write_sitexml", fake_write_sitexml)

        station_site = self._minimal_sera_site(
            "XX.ABCD", resource_id="quakeml:domain.ab/site/001")
        non_station_site = self._minimal_sera_site(
            None, resource_id="quakeml:domain.ab/site/without_station")

        sitedict_to_sitexml({
            "quakeml:domain.ab/site/001": station_site,
            "quakeml:domain.ab/site/without_station": non_station_site,
        }, output_folder=tmp_path)

        assert [tmp_path / "XX.ABCD.xml",
                tmp_path / "quakeml_domain_ab_site_without_station.xml"] == [
                    Path(filename) for _, filename, _ in output_calls]
        assert [validate for _, _, validate in output_calls] == [True, True]

    def test_sitexml_to_sitedict_reads_single_file(self, tmp_path):
        sera_site = self._minimal_sera_site()
        filename = tmp_path / "site.xml"
        write_sitexml(sera_site, filename)

        sera_site_dict = sitexml_to_sitedict(filename)

        assert list(sera_site_dict) == ["quakeml:domain.ab/site/001"]
        assert sera_site_dict["quakeml:domain.ab/site/001"].resource_id == (
            "quakeml:domain.ab/site/001")

    def test_sitexml_to_sitedict_reads_directory(self, tmp_path):
        site_001 = self._minimal_sera_site(
            "XX.ABCD", resource_id="quakeml:domain.ab/site/001")
        site_002 = self._minimal_sera_site(
            "YY.EFGH", resource_id="quakeml:domain.ab/site/002")
        write_sitexml(site_002, tmp_path / "b.xml")
        write_sitexml(site_001, tmp_path / "a.xml")

        sera_site_dict = sitexml_to_sitedict(tmp_path)

        assert list(sera_site_dict) == [
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
        ]

    def test_sitexml_to_sitedict_rejects_duplicate_site_ids(self, tmp_path):
        site_001 = self._minimal_sera_site(
            "XX.ABCD", resource_id="quakeml:domain.ab/site/001")
        site_002 = self._minimal_sera_site(
            "YY.EFGH", resource_id="quakeml:domain.ab/site/001")
        write_sitexml(site_001, tmp_path / "a.xml")
        write_sitexml(site_002, tmp_path / "b.xml")

        with pytest.raises(
                SiteXMLValidationError, match="Duplicate SiteXML site"):
            sitexml_to_sitedict(tmp_path)

    def test_sitexml_to_sitedict_rejects_missing_path(self, tmp_path):
        with pytest.raises(SiteXMLIOError):
            sitexml_to_sitedict(tmp_path / "missing.xml")

    def test_get_preferred_analysis(self):
        """
        Preferred analysis lookup follows the SiteDescription reference.
        """
        sera_site = self._minimal_sera_site()
        analysis_001 = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id)
        analysis_002 = Analysis(
            resource_id="quakeml:domain.ab/analysis/002",
            site_descriptionID=sera_site.site_description.resource_id)
        sera_site.analysis = [analysis_001, analysis_002]

        assert sera_site.get_preferred_analysis() is analysis_001

        sera_site.site_description.preferred_site_analysisID = (
            analysis_002.resource_id)
        assert sera_site.get_preferred_analysis() is analysis_002

        sera_site.site_description.preferred_site_analysisID = (
            "quakeml:domain.ab/analysis/missing")
        assert sera_site.get_preferred_analysis() is None

    def test_get_preferred_velocity_profile(self):
        """
        Preferred velocity profile lookup follows SiteDescription references.
        """
        sera_site = self._minimal_sera_site()
        profile_data = [VelocityProfileData(
            velocityS=ValueWithUncertainty(100),
            top_depth=ValueWithUncertainty(0))]
        profile_001 = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/001",
            velocity_profile_data=profile_data)
        profile_002 = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/002",
            velocity_profile_data=profile_data)
        profile_003 = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/003",
            velocity_profile_data=profile_data)
        analysis_001 = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[profile_001, profile_002]))
        analysis_002 = Analysis(
            resource_id="quakeml:domain.ab/analysis/002",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[profile_003]))
        sera_site.analysis = [analysis_001, analysis_002]

        assert sera_site.get_preferred_velocity_profile() is profile_001

        sera_site.site_description.preferred_velocity_profileID = (
            profile_002.resource_id)
        assert sera_site.get_preferred_velocity_profile() is profile_002

        sera_site.site_description.preferred_site_analysisID = (
            analysis_002.resource_id)
        sera_site.site_description.preferred_velocity_profileID = None
        assert sera_site.get_preferred_velocity_profile() is profile_003

        sera_site.site_description.preferred_velocity_profileID = (
            "quakeml:domain.ab/velocity_profile/missing")
        assert sera_site.get_preferred_velocity_profile() is None

    def test_get_indicator_object(self, testdata):
        """
        Indicator lookup uses site description and preferred analysis context.
        """
        sera_site = read_sitexml(testdata["full_sitexml.xml"])

        assert sera_site.get_indicator_object("siteClassEC8") is (
            sera_site.site_description.ec8)
        assert sera_site.get_indicator_object("velocityS30") is (
            sera_site.get_preferred_analysis().velocity_s30)
        assert sera_site.get_indicator_object("velocityProfileSet") is (
            sera_site.get_preferred_analysis().velocity_profile_set)

        with pytest.raises(
                SiteXMLValidationError,
                match="Unknown site indicator name"):
            sera_site.get_indicator_object("unknownIndicator")

    def test_add_site_indicator_routes_by_name(self):
        """
        Site indicators are assigned to their schema object location.
        """
        sera_site = self._minimal_sera_site()
        analysis_001 = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id)
        analysis_002 = Analysis(
            resource_id="quakeml:domain.ab/analysis/002",
            site_descriptionID=sera_site.site_description.resource_id)
        sera_site.analysis = [analysis_001, analysis_002]
        sera_site.site_description.preferred_site_analysisID = (
            analysis_002.resource_id)
        ec8 = EC8("A")
        velocity_s30 = VelocityS30(ValueWithUncertainty(760))

        sera_site.add_site_indicator([ec8, velocity_s30])

        assert sera_site.site_description.ec8 is ec8
        assert analysis_001.velocity_s30 is None
        assert analysis_002.velocity_s30 is velocity_s30

    def test_add_site_indicator_uses_explicit_analysis_id(self):
        """
        analysisID selects the target for analysis-level indicators.
        """
        sera_site = self._minimal_sera_site()
        analysis_001 = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id)
        analysis_002 = Analysis(
            resource_id=ResourceIdentifier(
                "quakeml:domain.ab/analysis/002"),
            site_descriptionID=sera_site.site_description.resource_id)
        sera_site.analysis = [analysis_001, analysis_002]
        velocity_s30 = VelocityS30(ValueWithUncertainty(760))

        sera_site.add_site_indicator(
            [velocity_s30], analysisID=analysis_002.resource_id)

        assert analysis_001.velocity_s30 is None
        assert analysis_002.velocity_s30 is velocity_s30

    def test_add_site_indicator_rejects_unknown_analysis_id(self):
        """
        Explicit analysisID must match an attached analysis.
        """
        sera_site = self._minimal_sera_site()
        sera_site.analysis = [Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id)]

        with pytest.raises(
                SiteXMLValidationError,
                match="analysisID does not match"):
            sera_site.add_site_indicator(
                [VelocityS30(ValueWithUncertainty(760))],
                analysisID="quakeml:domain.ab/analysis/missing")

    def test_add_site_indicator_requires_analysis_for_analysis_indicators(self):
        """
        Analysis-level indicators need an attached analysis target.
        """
        sera_site = self._minimal_sera_site()

        with pytest.raises(
                SiteXMLValidationError,
                match="without an attached analysis"):
            sera_site.add_site_indicator([
                VelocityS30(ValueWithUncertainty(760))])

    def _velocity_profile(self, resource_id):
        profile_data = [VelocityProfileData(
            velocityS=ValueWithUncertainty(100),
            top_depth=ValueWithUncertainty(0))]
        return VelocityProfile(
            resource_id=resource_id,
            velocity_profile_data=profile_data)

    def test_add_velocity_profiles_creates_velocity_profile_set(self):
        """
        Velocity profiles can be added to an analysis without an existing set.
        """
        sera_site = self._minimal_sera_site()
        analysis = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id)
        sera_site.analysis = [analysis]
        profile = self._velocity_profile(
            "quakeml:domain.ab/velocity_profile/new")

        velocity_profile_set = sera_site.add_velocity_profiles(
            [profile], analysisID=analysis.resource_id)

        assert analysis.velocity_profile_set is velocity_profile_set
        assert velocity_profile_set.velocity_profiles == [profile]

    def test_add_velocity_profiles_appends_and_rejects_duplicates(self):
        """
        Appending profiles preserves site-wide velocity profile ID uniqueness.
        """
        sera_site = self._minimal_sera_site()
        existing_profile = self._velocity_profile(
            "quakeml:domain.ab/velocity_profile/existing")
        analysis = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[existing_profile]))
        sera_site.analysis = [analysis]
        new_profile = self._velocity_profile(
            "quakeml:domain.ab/velocity_profile/new")

        sera_site.add_velocity_profiles([new_profile])

        assert analysis.velocity_profile_set.velocity_profiles == [
            existing_profile, new_profile]

        with pytest.raises(
                SiteXMLValidationError,
                match="Duplicate velocity profile resource_id"):
            sera_site.add_velocity_profiles([new_profile])

    def test_add_velocity_profiles_can_replace_existing_profiles(self):
        """
        Replacement updates the profile list but keeps set-level metadata.
        """
        sera_site = self._minimal_sera_site()
        existing_profile = self._velocity_profile(
            "quakeml:domain.ab/velocity_profile/existing")
        analysis = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[existing_profile],
                quality_index=0.5))
        sera_site.analysis = [analysis]
        new_profile = self._velocity_profile(
            "quakeml:domain.ab/velocity_profile/new")

        velocity_profile_set = sera_site.add_velocity_profiles(
            [new_profile], replace_existing=True)

        assert velocity_profile_set is analysis.velocity_profile_set
        assert velocity_profile_set.quality_index == 0.5
        assert velocity_profile_set.velocity_profiles == [new_profile]

    def test_validate_references_requires_preferred_velocity_profile_in_preferred_analysis(self):
        """
        Preferred velocity profile must belong to the preferred analysis.
        """
        sera_site = self._minimal_sera_site()
        profile_data = [VelocityProfileData(
            velocityS=ValueWithUncertainty(100),
            top_depth=ValueWithUncertainty(0))]
        profile_001 = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/001",
            velocity_profile_data=profile_data)
        profile_002 = VelocityProfile(
            resource_id="quakeml:domain.ab/velocity_profile/002",
            velocity_profile_data=profile_data)
        analysis_001 = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[profile_001]))
        analysis_002 = Analysis(
            resource_id="quakeml:domain.ab/analysis/002",
            site_descriptionID=sera_site.site_description.resource_id,
            velocity_profile_set=VelocityProfileSet(
                velocity_profiles=[profile_002]))
        sera_site.analysis = [analysis_001, analysis_002]
        sera_site.site_description.preferred_site_analysisID = (
            analysis_001.resource_id)
        sera_site.site_description.preferred_velocity_profileID = (
            profile_002.resource_id)

        with pytest.raises(
                SiteXMLValidationError,
                match="preferred_velocity_profileID does not belong"):
            sera_site.validate_references()

    def test_site_indicator_calculates_quality_index1(self, testdata):
        """
        Site indicators can calculate and store their Q_Index1 value.
        """
        sera_site = read_sitexml(testdata["full_sitedescription.xml"])
        ec8 = sera_site.site_description.ec8

        value = ec8.calculate_quality_index1(
            method="documented",
            evaluation="direct",
            reliability="partial",
            report="yes")

        assert value == 0.875
        assert ec8.quality_index == 0.875

    def test_site_indicator_quality_index_validates_range(self):
        """
        Quality indexes are optional numeric values in the closed [0, 1] range.
        """
        ec8 = EC8("A", quality_index="1")
        assert ec8.quality_index == 1.0

        ec8.quality_index = 0
        assert ec8.quality_index == 0.0

        ec8.quality_index = None
        assert ec8.quality_index is None

        for value in (-0.1, 1.1, "not-a-number", True):
            with pytest.raises(SiteXMLValidationError, match="quality_index"):
                ec8.quality_index = value

    def test_sera_site_calculates_quality_indexes(self, testdata):
        """
        SERASite exposes convenience methods for Q2, Q3, and overall QI.
        """
        sera_site = read_sitexml(testdata["full_sitexml.xml"])

        q2 = sera_site.calculate_quality_index2()
        q3 = sera_site.calculate_quality_index3(
            f0_vs30=1,
            f0_bedrock_depth=1,
            f0_h800=1,
            vs30_h800=1,
            vs30_geology=1)
        overall = sera_site.calculate_overall_quality_index(
            f0_vs30=1,
            f0_bedrock_depth=1,
            f0_h800=1,
            vs30_h800=1,
            vs30_geology=1)

        assert q2 == pytest.approx(3.04 / 4.25)
        assert q3 == 1
        assert overall == pytest.approx((q2 + q3) / 2)
        assert sera_site.site_description.overall_quality_index == overall

    def test_sera_site_quality_index3_uses_provided_consistency_pairs(
            self, testdata):
        """
        Q_Index3 averages only consistency pairs that are provided.
        """
        sera_site = read_sitexml(testdata["full_sitexml.xml"])

        q3 = sera_site.calculate_quality_index3(
            f0_vs30=1,
            f0_bedrock_depth=0)

        assert q3 == 0.5

    def test_sera_site_overall_quality_index_treats_missing_q3_as_zero(
            self, testdata):
        """
        Missing Q_Index3 is zero for the overall quality-index formula.
        """
        sera_site = read_sitexml(testdata["full_sitexml.xml"])

        q2 = sera_site.calculate_quality_index2()
        overall = sera_site.calculate_overall_quality_index()

        assert sera_site.calculate_quality_index3() is None
        assert overall == pytest.approx(q2 / 2)
        assert sera_site.site_description.overall_quality_index == overall

    def test_sera_site_overall_quality_index_is_zero_when_q2_is_zero(self):
        """
        Overall quality index is zero when Q_Index2 is zero.
        """
        sera_site = self._minimal_sera_site()

        value = sera_site.calculate_overall_quality_index(
            f0_vs30=1,
            f0_bedrock_depth=1,
            f0_h800=1,
            vs30_h800=1,
            vs30_geology=1)

        assert value == 0
        assert sera_site.site_description.overall_quality_index == 0

    def test_overall_quality_index_is_zero_when_q2_is_zero(self):
        """
        Formula helper returns zero when Q_Index2 is zero.
        """
        assert overall_quality_index(0, 1) == 0

    def test_add_sitexml_reference(self):
        """
        SiteXML URL is added as StationXML station ExternalReference.
        """
        station = Station(
            code="ABCD", latitude=1.0, longitude=2.0, elevation=3.0)
        inventory = Inventory(
            networks=[Network(code="XX", stations=[station])],
            source="TEST")

        returned = add_sitexml_reference(
            inventory,
            "XX.ABCD",
            "https://example.org/site.xml",
            added_time=obspy.UTCDateTime(2026, 5, 2, 12, 0, 0))

        assert returned is inventory
        refs = station.external_references
        assert len(refs) == 1
        assert refs[0].uri == "https://example.org/site.xml"
        assert refs[0].description == (
            "SERA SiteXML site characterization; added 2026-05-02")

    def test_add_sitexml_reference_replaces_managed_reference(self):
        """
        Existing helper-written SiteXML references are kept current.
        """
        station = Station(
            code="ABCD", latitude=1.0, longitude=2.0, elevation=3.0)
        station.external_references = [
            ExternalReference(
                uri="https://example.org/sitexml/"
                    "Site_XX.ABCD_01-05-2026.xml",
                description="SERA SiteXML site characterization; "
                    "added 2026-05-01"),
            ExternalReference(
                uri="https://example.org/other.xml",
                description="Unrelated station metadata"),
        ]
        inventory = Inventory(
            networks=[Network(code="XX", stations=[station])],
            source="TEST")

        add_sitexml_reference(
            inventory,
            "XX.ABCD",
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
            added_time=obspy.UTCDateTime(2026, 5, 2, 12, 0, 0))

        refs = station.external_references
        assert [ref.uri for ref in refs] == [
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
            "https://example.org/other.xml",
        ]
        assert refs[0].description == (
            "SERA SiteXML site characterization; added 2026-05-02")

    def test_add_sitexml_reference_replaces_manual_station_filename(
            self):
        """
        Manual references using the default SiteXML filename are replaced.
        """
        station = Station(
            code="ABCD", latitude=1.0, longitude=2.0, elevation=3.0)
        station.external_references = [
            ExternalReference(
                uri="https://example.org/sitexml/"
                    "Site_XX.ABCD_01-05-2026.xml",
                description="Manually added SiteXML link"),
            ExternalReference(
                uri="https://example.org/sitexml/"
                    "Site_YY.ABCD_01-05-2026.xml",
                description="Different station SiteXML link"),
        ]
        inventory = Inventory(
            networks=[Network(code="XX", stations=[station])],
            source="TEST")

        add_sitexml_reference(
            inventory,
            "XX.ABCD",
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
            added_time=obspy.UTCDateTime(2026, 5, 2, 12, 0, 0))

        assert [ref.uri for ref in station.external_references] == [
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
            "https://example.org/sitexml/Site_YY.ABCD_01-05-2026.xml",
        ]
        assert station.external_references[0].description == (
            "SERA SiteXML site characterization; added 2026-05-02")

    def test_add_sitexml_reference_can_append_history(self):
        """
        Replacement can be disabled when callers want to keep SiteXML history.
        """
        station = Station(
            code="ABCD", latitude=1.0, longitude=2.0, elevation=3.0)
        station.external_references = [
            ExternalReference(
                uri="https://example.org/sitexml/"
                    "Site_XX.ABCD_01-05-2026.xml",
                description="SERA SiteXML site characterization; "
                    "added 2026-05-01"),
        ]
        inventory = Inventory(
            networks=[Network(code="XX", stations=[station])],
            source="TEST")

        add_sitexml_reference(
            inventory,
            "XX.ABCD",
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
            added_time=obspy.UTCDateTime(2026, 5, 2, 12, 0, 0),
            replace_existing=False)

        assert [ref.uri for ref in station.external_references] == [
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml",
            "https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        ]

    def test_station_code_requires_network_station_notation(self):
        """
        Bare station codes are rejected to avoid ambiguous StationXML links.
        """
        valid_site = self._minimal_sera_site(station_code="1.ABC")
        assert valid_site.site_description.station_code == "1.ABC"

        invalid_codes = [
            "ABCD",
            "XXX.ABCD",
            "X.AB",
            "X.ABCDEF",
            "X.ABC1",
            "X.AB CD",
        ]
        for station_code in invalid_codes:
            with pytest.raises(SiteXMLValidationError, match="network.station"):
                self._minimal_sera_site(station_code=station_code)

    def test_station_code_schema_rejects_invalid_notation(self, testdata):
        """
        The SiteXML schema rejects the same invalid station notation.
        """
        xml = testdata["full_sitedescription.xml"].read_text(
            encoding="utf-8")
        xml = xml.replace("<station>XX.ABCD</station>",
                          "<station>XXX.ABCD</station>")

        with pytest.raises(SiteXMLValidationError):
            read_sitexml(io.BytesIO(xml.encode("utf-8")))

    def test_schema_accepts_revision_history(self, testdata):
        """
        The SiteXML schema accepts root-level document revision history.
        """
        xml = testdata["minimal_sitexml.xml"].read_text(encoding="utf-8")
        revision_history = (
            """
    <revisionHistory>
        <revision>
            <revisionTime>2026-05-02T12:00:00Z</revisionTime>
            <description>Updated velocity profile and quality indexes.</description>
            <author>ORFEUS</author>
            <version>2026-05-02</version>
            <previousVersion>"""
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml"
            """</previousVersion>
        </revision>
    </revisionHistory>"""
        )
        xml = xml.replace(
            "    <siteOwner>",
            revision_history + "\n    <siteOwner>",
            1)

        valid, errors = sitexml_module.validate_sitexml(
            io.BytesIO(xml.encode("utf-8")))

        assert valid
        assert errors == ()

    def test_add_sitexml_reference_rejects_missing_station(self):
        """
        The inventory must contain the exact network.station code.
        """
        inventory = Inventory(networks=[
            Network(code="YY", stations=[
                Station(
                    code="ABCD", latitude=1.0, longitude=2.0,
                    elevation=3.0)
            ]),
        ], source="TEST")

        with pytest.raises(SiteXMLValidationError, match="XX.ABCD"):
            add_sitexml_reference(
                inventory,
                "XX.ABCD",
                "https://example.org/site.xml")

    def test_value_with_uncertainty_to_float_with_uncertainties(self):
        """
        SiteXML symmetric uncertainty maps to both ObsPy uncertainty sides.
        """
        value = ValueWithUncertainty(12.5, uncertainty=0.4)

        converted = value.to_float_with_uncertainties()

        assert isinstance(converted, FloatWithUncertainties)
        assert float(converted) == 12.5
        assert converted.lower_uncertainty == 0.4
        assert converted.upper_uncertainty == 0.4

    def test_value_with_uncertainty_from_float_with_uncertainties(self):
        """
        Symmetric ObsPy uncertainty can be represented as SiteXML uncertainty.
        """
        value = FloatWithUncertainties(
            12.5, lower_uncertainty=0.4, upper_uncertainty=0.4)

        converted = ValueWithUncertainty.from_float_with_uncertainties(value)

        assert converted.value == 12.5
        assert converted.uncertainty == 0.4

    def test_value_with_uncertainty_rejects_asymmetric_uncertainty(self):
        """
        Reject asymmetric ObsPy uncertainty instead of losing one side.
        """
        value = FloatWithUncertainties(
            12.5, lower_uncertainty=0.3, upper_uncertainty=0.4)

        with pytest.raises(SiteXMLValidationError, match="symmetric"):
            ValueWithUncertainty.from_float_with_uncertainties(value)

    def test_site_owner_to_person(self):
        """
        SiteXML contact metadata maps to ObsPy Person lists.
        """
        site_owner = SERASiteOwner(
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name",
            person_firstname="Name",
            person_lastname="Surname",
            person_mbox="someemail@domain.ab",
            institution_name="INSTITUTION_ABBR")

        person = site_owner.to_person()

        assert isinstance(person, Person)
        assert person.names == ["Name Surname"]
        assert person.agencies == ["INSTITUTION_ABBR"]
        assert person.emails == ["someemail@domain.ab"]

    def test_site_owner_to_operator(self):
        """
        SiteXML owner metadata maps to an ObsPy Operator with one contact.
        """
        site_owner = SERASiteOwner(
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name",
            person_firstname="Name",
            person_lastname="Surname",
            person_mbox="someemail@domain.ab",
            institution_homepage="https://www.domain.ab")

        operator = site_owner.to_operator()

        assert isinstance(operator, Operator)
        assert operator.agency == "Site Owner Full Name"
        assert operator.website == "https://www.domain.ab"
        assert len(operator.contacts) == 1
        assert operator.contacts[0].names == ["Name Surname"]

    def test_site_owner_from_person(self):
        """
        ObsPy Person metadata can seed SiteXML owner contact metadata.
        """
        person = Person(
            names=["Name Surname"],
            agencies=["INSTITUTION_ABBR"],
            emails=["someemail@domain.ab"])

        site_owner = SERASiteOwner.from_person(
            person,
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name")

        assert site_owner.owner_codename == "SITEOWNER"
        assert site_owner.owner_fullname == "Site Owner Full Name"
        assert site_owner.person_firstname == "Name"
        assert site_owner.person_lastname == "Surname"
        assert site_owner.person_mbox == "someemail@domain.ab"
        assert site_owner.institution_name == "INSTITUTION_ABBR"

    def test_site_owner_from_operator(self):
        """
        ObsPy Operator metadata can seed SiteXML owner contact metadata.
        """
        operator = Operator(
            agency="Site Owner Full Name",
            contacts=[Person(
                names=["Name Surname"],
                emails=["someemail@domain.ab"])],
            website="https://www.domain.ab")

        site_owner = SERASiteOwner.from_operator(
            operator, owner_codename="SITEOWNER")

        assert site_owner.owner_codename == "SITEOWNER"
        assert site_owner.owner_fullname == "Site Owner Full Name"
        assert site_owner.person_firstname == "Name"
        assert site_owner.person_lastname == "Surname"
        assert site_owner.person_mbox == "someemail@domain.ab"
        assert site_owner.institution_homepage == "https://www.domain.ab"

    def test_site_owner_from_operator_rejects_ambiguous_contacts(self):
        """
        Reject multiple ObsPy contacts unless the caller selects one.
        """
        operator = Operator(
            agency="Site Owner Full Name",
            contacts=[
                Person(names=["Name Surname"], emails=["one@domain.ab"]),
                Person(names=["Other Contact"], emails=["two@domain.ab"])])

        with pytest.raises(SiteXMLValidationError, match="multiple contacts"):
            SERASiteOwner.from_operator(
                operator, owner_codename="SITEOWNER")

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

        def normalize_root_creation_time(lines):
            normalized = list(lines)
            for index, line in enumerate(normalized):
                if "<creationTime>" in line:
                    normalized[index] = re.sub(
                        r"<creationTime>.*</creationTime>",
                        "<creationTime>IGNORED</creationTime>",
                        line,
                        count=1)
                    break
            return normalized

        new_lines = normalize_root_creation_time(new_lines)
        org_lines = normalize_root_creation_time(org_lines)

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
        sitexmls = [
            testdata["full_sitexml.xml"],
            testdata["full_sitedescription_without_station.xml"],
        ]
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

    def test_read_sitexml_accepts_http_url(self, testdata, monkeypatch):
        filename = testdata["minimal_sitexml.xml"]

        with open(filename, "rb") as fh:
            xml = fh.read()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return xml

        def fake_urlopen(url, timeout):
            assert url == "https://example.org/site.xml"
            assert timeout == 30
            return FakeResponse()

        monkeypatch.setattr(sitexml_module, "urlopen", fake_urlopen)

        sera_site = read_sitexml("https://example.org/site.xml")

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

    def test_resource_id_fields_normalize_resourceidentifier_inputs(self):
        site_owner = SERASiteOwner(
            owner_codename="TEST",
            owner_fullname="Test Owner",
            person_firstname="Name",
            person_lastname="Surname",
            person_mbox="someemail@domain.ab",
            ownerID=ResourceIdentifier("quakeml:domain.ab/siteOwner/001"),
            personID=ResourceIdentifier("quakeml:domain.ab/person/001"),
            institutionID=ResourceIdentifier("quakeml:domain.ab/institution/001"),
        )
        site_description = SiteDescription(
            resource_id=ResourceIdentifier(
                "quakeml:domain.ab/site_description/001"),
            latitude=1.0,
            longitude=2.0,
            preferred_site_analysisID=ResourceIdentifier(
                "quakeml:domain.ab/analysis/001"),
            preferred_velocity_profileID=ResourceIdentifier(
                "quakeml:domain.ab/velocity_profile/001"),
        )
        analysis = Analysis(
            resource_id=ResourceIdentifier("quakeml:domain.ab/analysis/001"),
            site_descriptionID=ResourceIdentifier(
                "quakeml:domain.ab/site_description/001"),
        )
        velocity_profile = VelocityProfile(
            resource_id=ResourceIdentifier(
                "quakeml:domain.ab/velocity_profile/001"),
            velocity_profile_data=[VelocityProfileData(
                velocityS=ValueWithUncertainty(100),
                top_depth=ValueWithUncertainty(0))
            ],
        )
        sera_site = SERASite(
            resource_id=ResourceIdentifier("quakeml:domain.ab/site/001"),
            site_owner=site_owner,
            site_description=site_description,
            analysis=[analysis],
        )

        assert isinstance(sera_site.resource_id, str)
        assert sera_site.resource_id == "quakeml:domain.ab/site/001"
        assert site_owner.ownerID == "quakeml:domain.ab/siteOwner/001"
        assert site_owner.personID == "quakeml:domain.ab/person/001"
        assert site_owner.institutionID == "quakeml:domain.ab/institution/001"
        assert site_description.resource_id == (
            "quakeml:domain.ab/site_description/001")
        assert site_description.preferred_site_analysisID == (
            "quakeml:domain.ab/analysis/001")
        assert site_description.preferred_velocity_profileID == (
            "quakeml:domain.ab/velocity_profile/001")
        assert analysis.resource_id == "quakeml:domain.ab/analysis/001"
        assert analysis.site_descriptionID == (
            "quakeml:domain.ab/site_description/001")
        assert velocity_profile.resource_id == (
            "quakeml:domain.ab/velocity_profile/001")

    def test_sera_site_get_analysis_by_resource_id(self):
        site_description = SiteDescription(
            resource_id="quakeml:domain.ab/site_description/001",
            latitude=1.0,
            longitude=2.0)
        analysis = Analysis(
            resource_id="quakeml:domain.ab/analysis/001",
            site_descriptionID="quakeml:domain.ab/site_description/001")
        sera_site = SERASite(
            resource_id="quakeml:domain.ab/site/001",
            site_owner=SERASiteOwner(
                owner_codename="TEST",
                owner_fullname="Test Owner",
                person_firstname="Name",
                person_lastname="Surname",
                person_mbox="someemail@domain.ab"),
            site_description=site_description,
            analysis=[analysis])

        assert sera_site.get_analysis(
            "quakeml:domain.ab/analysis/001") is analysis
        assert sera_site.get_analysis(
            ResourceIdentifier("quakeml:domain.ab/analysis/001")) is analysis
        assert sera_site.get_analysis(
            "quakeml:domain.ab/analysis/missing") is None

    def test_sitexml_created_validates_utcdatetime(self):
        with pytest.raises(SiteXMLValidationError):
            SERASite(
                resource_id="quakeml:domain.ab/site/001",
                site_owner=SERASiteOwner(
                    owner_codename="TEST",
                    owner_fullname="Test Owner",
                    person_firstname="Name",
                    person_lastname="Surname",
                    person_mbox="someemail@domain.ab"),
                site_description=SiteDescription(
                    resource_id="quakeml:domain.ab/site_description/001",
                    latitude=1.0,
                    longitude=2.0),
                created=object())

    def test_revision_requires_schema_required_fields(self):
        """
        Revision objects require a time and non-empty description.
        """
        revision = Revision(
            revision_time="2026-05-02T12:00:00Z",
            description="Updated velocity profile.",
            author="ORFEUS",
            version=20260502,
            previous_version=(
                "https://example.org/sitexml/"
                "Site_XX.ABCD_01-05-2026.xml"))

        assert revision.revision_time == obspy.UTCDateTime(
            2026, 5, 2, 12, 0, 0)
        assert revision.description == "Updated velocity profile."
        assert revision.author == "ORFEUS"
        assert revision.version == "20260502"
        assert revision.previous_version == (
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml")

        with pytest.raises(SiteXMLValidationError):
            Revision(None, "Updated velocity profile.")

        with pytest.raises(SiteXMLValidationError):
            Revision("2026-05-02T12:00:00Z", "")

    def test_revision_history_round_trips(self):
        """
        Root-level revision history is written and read as Revision objects.
        """
        sera_site = self._minimal_sera_site(station_code="XX.ABCD")
        sera_site.revision_history = [
            Revision(
                revision_time=obspy.UTCDateTime(2026, 5, 2, 12, 0, 0),
                description="Updated velocity profile and quality indexes.",
                author="ORFEUS",
                version="2026-05-02",
                previous_version=(
                    "https://example.org/sitexml/"
                    "Site_XX.ABCD_01-05-2026.xml")),
        ]

        xml_buffer = io.BytesIO()
        write_sitexml(sera_site, xml_buffer, validate=True)
        xml_buffer.seek(0)

        reread = read_sitexml(xml_buffer)

        assert len(reread.revision_history) == 1
        revision = reread.revision_history[0]
        assert revision.revision_time == obspy.UTCDateTime(
            2026, 5, 2, 12, 0, 0)
        assert revision.description == (
            "Updated velocity profile and quality indexes.")
        assert revision.author == "ORFEUS"
        assert revision.version == "2026-05-02"
        assert revision.previous_version == (
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml")

    def test_sera_site_add_revision(self):
        """
        SERASite.add_revision appends and returns a Revision object.
        """
        sera_site = self._minimal_sera_site(station_code="XX.ABCD")

        revision = sera_site.add_revision(
            revision_time="2026-05-02T12:00:00Z",
            description="Updated velocity profile.",
            author="ORFEUS",
            version="2026-05-02",
            previous_version=(
                "https://example.org/sitexml/"
                "Site_XX.ABCD_01-05-2026.xml"))

        assert sera_site.revision_history == [revision]
        assert revision.revision_time == obspy.UTCDateTime(
            2026, 5, 2, 12, 0, 0)
        assert revision.description == "Updated velocity profile."
        assert revision.author == "ORFEUS"
        assert revision.version == "2026-05-02"
        assert revision.previous_version == (
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml")

    def test_write_sitexml_uses_serialization_time(self):
        sera_site = SERASite(
            resource_id="quakeml:domain.ab/site/001",
            site_owner=SERASiteOwner(
                owner_codename="TEST",
                owner_fullname="Test Owner",
                person_firstname="Name",
                person_lastname="Surname",
                person_mbox="someemail@domain.ab"),
            site_description=SiteDescription(
                resource_id="quakeml:domain.ab/site_description/001",
                latitude=1.0,
                longitude=2.0),
            created=obspy.UTCDateTime(2000, 1, 1))

        before = obspy.UTCDateTime()
        xml_buffer = io.BytesIO()
        write_sitexml(sera_site, xml_buffer, validate=True)
        after = obspy.UTCDateTime()

        root = etree.fromstring(xml_buffer.getvalue())
        written_creation_time = obspy.UTCDateTime(
            root.find("{http://www.orfeus-eu.org/xml/site/1}creationTime").text)

        assert before <= written_creation_time <= after
        assert sera_site.created == written_creation_time

    def test_write_sitexml_none_uses_default_filename(
            self, tmp_path, monkeypatch):
        sera_site = self._minimal_sera_site(station_code="XX.ABCD")
        monkeypatch.chdir(tmp_path)

        before = obspy.UTCDateTime()
        write_sitexml(sera_site, None, validate=True)
        after = obspy.UTCDateTime()

        filename = tmp_path / sera_site.get_sitexml_filename(sera_site.created)
        assert filename.exists()

        root = etree.parse(str(filename)).getroot()
        written_creation_time = obspy.UTCDateTime(
            root.find("{http://www.orfeus-eu.org/xml/site/1}creationTime").text)

        assert before <= written_creation_time <= after
        assert sera_site.created == written_creation_time
        assert filename.name == (
            "Site_XX.ABCD_%s.xml" %
            written_creation_time.strftime("%d-%m-%Y"))

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

    def test_enum_list_property_validates_list_mutations(self):
        resonance_frequency = ResonanceFrequency(
            value=ValueWithUncertainty(1.0),
            methods=["hvsr noise"])

        assert resonance_frequency.methods == ["HVSR NOISE"]

        resonance_frequency.methods.append("ssr earthquake records")
        resonance_frequency.methods.insert(0, "inferred")

        assert resonance_frequency.methods == [
            "INFERRED",
            "HVSR NOISE",
            "SSR EARTHQUAKE RECORDS",
        ]

        with pytest.raises(SiteXMLValidationError):
            resonance_frequency.methods.append("not-a-method")

        with pytest.raises(SiteXMLValidationError):
            resonance_frequency.methods.insert(0, 123)

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

    def test_write_sitexml_validates_analysis_site_description_reference(
            self, testdata):
        sera_site = read_sitexml(testdata["full_analysis.xml"])
        sera_site.analysis[0].site_descriptionID = (
            "quakeml:domain.ab/site_description/does-not-match")

        with pytest.raises(SiteXMLValidationError, match="site_descriptionID"):
            write_sitexml(sera_site, io.BytesIO(), validate=True)

    def test_write_sitexml_validates_preferred_analysis_reference(
            self, testdata):
        sera_site = read_sitexml(testdata["full_analysis.xml"])
        sera_site.site_description.preferred_site_analysisID = (
            "quakeml:domain.ab/analysis/missing")

        with pytest.raises(
                SiteXMLValidationError, match="preferred_site_analysisID"):
            write_sitexml(sera_site, io.BytesIO(), validate=True)

    def test_write_sitexml_validates_preferred_velocity_profile_reference(
            self, testdata):
        sera_site = read_sitexml(testdata["full_analysis.xml"])
        sera_site.site_description.preferred_velocity_profileID = (
            "quakeml:domain.ab/velocity_profile/missing")

        with pytest.raises(
                SiteXMLValidationError, match="preferred_velocity_profileID"):
            write_sitexml(sera_site, io.BytesIO(), validate=True)

    def test_write_sitexml_validates_duplicate_analysis_ids(self, testdata):
        sera_site = read_sitexml(testdata["full_analysis.xml"])
        duplicate_analysis = sera_site.analysis[0].copy()
        sera_site.analysis.append(duplicate_analysis)

        with pytest.raises(
                SiteXMLValidationError, match="Duplicate analysis resource_id"):
            write_sitexml(sera_site, io.BytesIO(), validate=True)

    def test_write_sitexml_validates_duplicate_velocity_profile_ids(
            self, testdata):
        sera_site = read_sitexml(testdata["full_analysis.xml"])
        velocity_profiles = (
            sera_site.analysis[0].velocity_profile_set.velocity_profiles)
        velocity_profiles[1].resource_id = velocity_profiles[0].resource_id

        with pytest.raises(
                SiteXMLValidationError,
                match="Duplicate velocity profile resource_id"):
            write_sitexml(sera_site, io.BytesIO(), validate=True)
        
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
        assert sera_site.site_description.station_code == "XX.ABCD"
        assert sera_site.site_description.latitude == 45.137174
        assert sera_site.site_description.longitude == 5.998905

        assert sera_site.site_description.altitude == 239.0
        assert sera_site.site_description.min_distance_from_station == 10.3
        assert sera_site.site_description.max_distance_from_station == 10.3

        assert sera_site.site_description.topographyA == "T1"
        assert sera_site.site_description.topographyB == "Valley"
        assert sera_site.site_description.morphology == "Valley - Basin"

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

        assert sera_site.site_description.ec8.external_references is not None
        assert len(sera_site.site_description.ec8.external_references) == 2
        external_ref = sera_site.site_description.ec8.external_references[0]
        assert external_ref.uri == "https://doi.org/10.1007/s10518-017-0135-5/"
        assert external_ref.description == "paper"
        external_ref = sera_site.site_description.ec8.external_references[1]
        assert external_ref.uri == (
            "https://www.domain.ab/SiteXML/ec8-supporting-resource"
        )
        assert external_ref.description == "supporting resource"

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

    def test_reading_missing_site_indicator_quality_index_preserves_none(
            self, testdata):
        """
        Missing optional qualityIndex stays None in the object model.
        """
        filename = testdata["full_sitedescription.xml"]
        xml = filename.read_bytes().replace(
            b"                <qualityIndex>1.0</qualityIndex>\n",
            b"",
            1)

        sera_site = read_sitexml(io.BytesIO(xml))

        assert sera_site.site_description.ec8.quality_index is None

    def test_reading_missing_velocity_profile_quality_index_preserves_none(
            self, testdata):
        """
        Missing optional velocityProfileSet qualityIndex stays None.
        """
        filename = testdata["full_analysis.xml"]
        xml = filename.read_bytes().replace(
            b"            <qualityIndex>1.0</qualityIndex>\n",
            b"",
            1)

        sera_site = read_sitexml(io.BytesIO(xml))

        assert sera_site.analysis[0].velocity_profile_set.quality_index is None

    def test_reading_and_writing_full_sitedescription_without_station_tag(
            self, testdata):
        """
        Tests reading and writing a full SiteXML <siteDescription> tag for a
        site without a station installation.
        """
        filename = testdata["full_sitedescription_without_station.xml"]
        sera_site = read_sitexml(filename)

        assert sera_site.site_description is not None
        assert sera_site.site_description.resource_id == (
            "quakeml:domain.ab/site_description/003")
        assert sera_site.site_description.station_code is None
        assert sera_site.site_description.latitude == 40.555907
        assert sera_site.site_description.longitude == 22.988593
        assert sera_site.site_description.altitude == 120.0
        assert sera_site.site_description.min_distance_from_station is None
        assert sera_site.site_description.max_distance_from_station is None
        assert sera_site.site_description.topographyA == "T1"
        assert sera_site.site_description.topographyB == "Flat"
        assert sera_site.site_description.morphology == "Plain"

        assert sera_site.site_description.ec8 is not None
        assert sera_site.site_description.ec8.value == "A"
        assert sera_site.site_description.ec8.quality_index == 1.0

        assert sera_site.site_description.bedrock_depth is not None
        assert sera_site.site_description.bedrock_depth.value.value == 820.0
        assert sera_site.site_description.bedrock_depth.value.uncertainty is None
        assert sera_site.site_description.bedrock_depth.quality_index == 0.8

        assert sera_site.site_description.h800 is not None
        assert sera_site.site_description.h800.value.value == 180.0
        assert sera_site.site_description.h800.value.uncertainty is None

        assert sera_site.site_description.geological_unit is not None
        assert sera_site.site_description.geological_unit.value == (
            "Holocene Deposits")
        assert sera_site.site_description.geological_unit.geological_map_scale == (
            "1:50000")
        assert sera_site.site_description.preferred_site_analysisID is None
        assert sera_site.site_description.preferred_velocity_profileID is None

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

        assert f0.external_references is not None
        assert len(f0.external_references) == 1
        external_ref = f0.external_references[0]
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
        Tests reading and writing a full SiteXML <velocityProfileSet> tag.
        """
        filename = testdata["full_analysis.xml"]
        sera_site = read_sitexml(filename)

        assert sera_site.analysis[0] is not None
        assert sera_site.analysis[0].velocity_profile_set is not None

        vps = sera_site.analysis[0].velocity_profile_set
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
        assert vpd.bottom_depth is None

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
        assert vpd.bottom_depth is None

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
        vp = sera_site.analysis[0].velocity_profile_set.velocity_profiles[0]
        assert vp.layer_count == 8
        assert len(vp.velocity_profile_data) == 8

    def test_reading_velocity_profile_requires_velocity_s(
            self, testdata, tmp_path):
        """
        Tests that velocityProfileData requires velocityS.
        """
        filename = testdata["full_analysis.xml"]
        xml_text = filename.read_text(encoding="utf-8")
        invalid_xml = tmp_path / "missing_velocity_s.xml"
        invalid_xml.write_text(
            re.sub(
                r"\s*<velocityS>\s*<value>118\.08</value>\s*"
                r"<uncertainty>2\.0</uncertainty>\s*</velocityS>",
                "",
                xml_text,
                count=1),
            encoding="utf-8")

        with pytest.raises(SiteXMLValidationError, match="velocityS"):
            read_sitexml(invalid_xml)

    def test_velocity_profile_requires_schema_required_fields(self):
        """
        Tests required VelocityProfile and VelocityProfileData fields.
        """
        velocity_s = ValueWithUncertainty(100.0)
        top_depth = ValueWithUncertainty(0.0)
        layer = VelocityProfileData(velocityS=velocity_s, top_depth=top_depth)

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
            VelocityProfileData(velocityS=velocity_s, top_depth=None)

        with pytest.raises(SiteXMLValidationError, match="velocityS"):
            VelocityProfileData(velocityS=None, top_depth=top_depth)

    def test_velocity_profile_derives_layer_count_from_data(self):
        """
        Tests that layer_count is derived from velocity_profile_data when omitted.
        """
        layers = [
            VelocityProfileData(
                top_depth=ValueWithUncertainty(0.0),
                velocityS=ValueWithUncertainty(100.0)),
            VelocityProfileData(
                top_depth=ValueWithUncertainty(10.0),
                velocityS=ValueWithUncertainty(200.0)),
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
