#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for SiteXML CSV and Excel import helpers.
"""

from pathlib import Path
import warnings

import pandas as pd
import pytest

from obspy.io.sitexml import read_csv as read_csv_module
from obspy.io.sitexml.core import SERASite, SERASiteOwner, SiteDescription
from obspy.io.sitexml.util import SiteXMLIOError, SiteXMLImportError
from obspy.io.sitexml.read_csv import (csv_to_sera_site, excel_to_sera_site,
                                       _read_year_cell)


class TestSiteXMLCSVImport():
    def _minimal_sera_site(self, resource_id, station_code="XX.ABCD"):
        site_owner = SERASiteOwner(
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name",
            person_firstname="Name",
            person_lastname="Surname",
            person_mbox="someemail@domain.ab")
        site_description = SiteDescription(
            resource_id=resource_id + "/description",
            station_code=station_code,
            latitude=1.0,
            longitude=2.0)
        return SERASite(
            resource_id=resource_id,
            site_owner=site_owner,
            site_description=site_description)

    def test_sitedict_to_sitexml_uses_network_station_filename(
            self, tmp_path, monkeypatch):
        output_calls = []

        def fake_write_sitexml(sera_site, filename, validate):
            output_calls.append((sera_site, filename, validate))

        monkeypatch.setattr(
            read_csv_module, "write_sitexml", fake_write_sitexml)

        station_site = self._minimal_sera_site(
            "quakeml:domain.ab/site/001", station_code="XX.ABCD")
        non_station_site = self._minimal_sera_site(
            "quakeml:domain.ab/site/without_station", station_code=None)

        read_csv_module.sitedict_to_sitexml({
            "quakeml:domain.ab/site/001": station_site,
            "quakeml:domain.ab/site/without_station": non_station_site,
        }, output_folder=tmp_path)

        assert [tmp_path / "XX.ABCD.xml",
                tmp_path / "quakeml_domain_ab_site_without_station.xml"] == [
                    Path(filename) for _, filename, _ in output_calls]
        assert [validate for _, _, validate in output_calls] == [True, True]

    def test_read_year_cell_preserves_schema_string_type(self):
        assert _read_year_cell(2018) == "2018"
        assert _read_year_cell(2018.0) == "2018"
        assert _read_year_cell("2018") == "2018"

        with pytest.raises(SiteXMLImportError):
            _read_year_cell("18")

    def _assert_full_reference_metadata(self, site_indicator):
        literature_source = site_indicator.literature_source
        assert literature_source is not None
        assert literature_source.title == "Some title"
        assert literature_source.first_author == "Author A."
        assert literature_source.secondary_authors == "Author B., Author C."
        assert literature_source.year == "2018"
        assert literature_source.booktitle == "Some magazine"
        assert literature_source.language == "en"
        assert literature_source.doi == "10.1007/s10518-017-0135-5"

        external_references = site_indicator.external_references
        assert len(external_references) == 1
        external_reference = external_references[0]
        assert external_reference.uri == (
            "https://doi.org/10.1007/s10518-017-0135-5/")
        assert external_reference.description == "paper"

    def test_csv_to_sera_site_imports_sites_analysis_and_velocity_profiles(
            self, datapath):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=datapath / "site_owner.csv",
            site_description_csv=datapath / "site_description.csv",
            analysis_csv=datapath / "site_analysis.csv",
            velocity_profiles_csv=datapath / "velocity_profiles",
            delim=";")

        assert set(sera_site_dict) == {
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
            "quakeml:domain.ab/site/003",
        }

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        assert site_001.site_owner.owner_codename == "SITEOWNER"
        assert site_001.created is None
        assert site_001.site_description.resource_id == (
            "quakeml:domain.ab/site_description/001")
        assert site_001.site_description.station_code == "XX.ABCD"
        assert site_001.site_description.preferred_site_analysisID == (
            "quakeml:domain.ab/analysis/001")
        assert len(site_001.analysis) == 3

        analysis_001 = site_001.analysis[0]
        assert analysis_001.resource_id == "quakeml:domain.ab/analysis/001"
        assert analysis_001.site_descriptionID == (
            "quakeml:domain.ab/site_description/001")
        assert analysis_001.resonance_frequency.value.value == 0.7
        assert analysis_001.velocity_s30.value.value == 620.0
        assert analysis_001.velocity_s30.value.uncertainty == 18.0
        assert analysis_001.velocity_s30.methods == ["MASW", "SPAC/F-K"]
        assert analysis_001.velocity_profile_survey is not None
        assert len(analysis_001.velocity_profile_survey.velocity_profiles) == 2

        first_profile = analysis_001.velocity_profile_survey.velocity_profiles[0]
        assert first_profile.resource_id == (
            "quakeml:domain.ab/velocity_profile/001")
        assert first_profile.layer_count == 8
        assert len(first_profile.velocity_profile_data) == 8
        assert first_profile.velocity_profile_data[0].velocityS.value == 118.08
        assert first_profile.velocity_profile_data[0].velocityS.uncertainty == 2.0
        assert first_profile.velocity_profile_data[0].top_depth.value == 0.0
        assert first_profile.velocity_profile_data[0].bottom_depth.value == 0.19

        site_002 = sera_site_dict["quakeml:domain.ab/site/002"]
        assert site_002.site_description.resource_id == (
            "quakeml:domain.ab/site_description/002")
        assert site_002.site_description.station_code == "YY.WXYZ"
        assert len(site_002.analysis) == 1

        analysis_002 = site_002.analysis[0]
        assert analysis_002.resource_id == "quakeml:domain.ab/analysis/004"
        assert analysis_002.resonance_frequency.value.value == 0.3
        assert analysis_002.velocity_s30.value.value == 497.0
        assert analysis_002.velocity_s30.methods == ["S-REFL"]
        assert analysis_002.velocity_profile_survey is not None
        assert len(analysis_002.velocity_profile_survey.velocity_profiles) == 3
        assert site_002.site_description.h800.quality_index is None
        assert site_002.site_description.geological_unit.quality_index is None

        site_003 = sera_site_dict["quakeml:domain.ab/site/003"]
        assert site_003.site_description.resource_id == (
            "quakeml:domain.ab/site_description/003")
        assert site_003.site_description.station_code is None
        assert site_003.site_description.topographyA == "T2"
        assert site_003.site_description.topographyB == "Flat"
        assert site_003.site_description.morphology == "Plain"
        assert site_003.site_description.preferred_site_analysisID is None
        assert site_003.site_description.preferred_velocity_profileID is None
        assert site_003.analysis is None

    def test_csv_to_sera_site_imports_vs30_quality_indexes(self, datapath):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=datapath / "site_owner.csv",
            site_description_csv=datapath / "site_description.csv",
            analysis_csv=datapath / "site_analysis.csv",
            velocity_profiles_csv=datapath / "velocity_profiles",
            delim=";")

        analysis_001 = (
            sera_site_dict["quakeml:domain.ab/site/001"].analysis[0]
        )

        assert analysis_001.velocity_s30.method_combined_qindex == "1.2"
        assert analysis_001.velocity_s30.manual_qindex == "1.0"

    def test_csv_to_sera_site_imports_full_reference_metadata(self, datapath):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=datapath / "site_owner.csv",
            site_description_csv=datapath / "site_description.csv",
            analysis_csv=datapath / "site_analysis.csv",
            velocity_profiles_csv=datapath / "velocity_profiles",
            delim=";")

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        analysis_001 = site_001.analysis[0]

        self._assert_full_reference_metadata(site_001.site_description.ec8)
        self._assert_full_reference_metadata(analysis_001.resonance_frequency)
        self._assert_full_reference_metadata(
            analysis_001.velocity_profile_survey)

    def test_excel_to_sera_site_imports_sites_analysis_and_velocity_profiles(
            self, datapath):
        pytest.importorskip("openpyxl")

        sera_site_dict = excel_to_sera_site(
            path_or_file_object=datapath / "sera_site_all.xlsx",
            velocity_profiles=datapath / "velocity_profiles.xlsx")

        assert set(sera_site_dict) == {
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
        }

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        assert site_001.site_owner.owner_codename == "SITEOWNER"
        assert site_001.site_description.resource_id == (
            "quakeml:domain.ab/site_description/001")
        assert site_001.site_description.station_code == "XX.ABCD"
        assert site_001.site_description.preferred_site_analysisID == (
            "quakeml:domain.ab/analysis/001")
        assert len(site_001.analysis) == 3

        analysis_001 = site_001.analysis[0]
        assert analysis_001.resource_id == "quakeml:domain.ab/analysis/001"
        assert analysis_001.site_descriptionID == (
            "quakeml:domain.ab/site_description/001")
        assert analysis_001.resonance_frequency.value.value == 0.7
        assert analysis_001.velocity_s30.value.value == 620.0
        assert analysis_001.velocity_s30.value.uncertainty == 18.0
        assert analysis_001.velocity_s30.methods == ["MASW", "SPAC/F-K"]
        assert analysis_001.velocity_s30.method_combined_qindex == "1.2"
        assert analysis_001.velocity_s30.manual_qindex == "1.0"
        assert analysis_001.velocity_profile_survey is not None
        assert len(analysis_001.velocity_profile_survey.velocity_profiles) == 2

        first_profile = analysis_001.velocity_profile_survey.velocity_profiles[0]
        assert first_profile.resource_id == (
            "quakeml:domain.ab/velocity_profile/001")
        assert first_profile.layer_count == 8
        assert len(first_profile.velocity_profile_data) == 8
        assert first_profile.velocity_profile_data[0].velocityS.value == 118.08
        assert first_profile.velocity_profile_data[0].velocityS.uncertainty == 2.0
        assert first_profile.velocity_profile_data[0].top_depth.value == 0.0
        assert first_profile.velocity_profile_data[0].bottom_depth.value == 0.19

        site_002 = sera_site_dict["quakeml:domain.ab/site/002"]
        assert site_002.site_description.resource_id == (
            "quakeml:domain.ab/site_description/002")
        assert len(site_002.analysis) == 1

        analysis_002 = site_002.analysis[0]
        assert analysis_002.resource_id == "quakeml:domain.ab/analysis/004"
        assert analysis_002.resonance_frequency.value.value == 0.3
        assert analysis_002.velocity_s30.value.value == 497.0
        assert analysis_002.velocity_s30.methods == ["S-REFL"]
        assert analysis_002.velocity_profile_survey is not None
        assert len(analysis_002.velocity_profile_survey.velocity_profiles) == 3

    def test_excel_to_sera_site_imports_full_reference_metadata(self, datapath):
        pytest.importorskip("openpyxl")

        sera_site_dict = excel_to_sera_site(
            path_or_file_object=datapath / "sera_site_all.xlsx",
            velocity_profiles=datapath / "velocity_profiles.xlsx")

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        analysis_001 = site_001.analysis[0]

        self._assert_full_reference_metadata(site_001.site_description.ec8)
        self._assert_full_reference_metadata(analysis_001.resonance_frequency)
        self._assert_full_reference_metadata(
            analysis_001.velocity_profile_survey)

    def test_excel_to_sera_site_warns_when_analysis_sheet_is_missing(
            self, datapath):
        pytest.importorskip("openpyxl")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sera_site_dict = excel_to_sera_site(
                path_or_file_object=datapath / "sera_site_no_analysis.xlsx",
                velocity_profiles=datapath / "velocity_profiles.xlsx")

        assert set(sera_site_dict) == {
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
        }
        assert any("Analysis metadata not provided." in str(w.message)
                   for w in caught)

    def test_excel_to_sera_site_raises_for_missing_required_owner_sheet(
            self, datapath):
        pytest.importorskip("openpyxl")

        with pytest.raises(SiteXMLImportError):
            excel_to_sera_site(datapath / "sera_site_no_owner.xlsx")

    def test_excel_to_sera_site_raises_for_missing_required_site_description_sheet(
            self, datapath):
        pytest.importorskip("openpyxl")

        with pytest.raises(SiteXMLImportError):
            excel_to_sera_site(datapath / "sera_site_no_sd.xlsx")

    def test_csv_to_sera_site_skips_invalid_site_description_rows(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude;station\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0;XX.AAA\n"
            ";quakeml:test/site_description/002;46.0;8.0;XX.BBB\n"
            "quakeml:test/site/003;;47.0;9.0;XX.CCC\n",
            encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sera_site_dict = csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                delim=";")

        assert set(sera_site_dict) == {"quakeml:test/site/001"}
        assert any("Missing siteID, siteDescriptionID, latitude or longitude"
                   in str(w.message)
                   for w in caught)

    def test_csv_to_sera_site_raises_for_invalid_optional_analysis_rows(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;velocityS30_value\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;quakeml:test/site_description/001;300\n"
            "quakeml:test/site/001;;quakeml:test/site_description/001;250\n",
            encoding="utf-8")

        with pytest.raises(SiteXMLImportError):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                delim=";")

    def test_csv_to_sera_site_raises_for_incomplete_literature_source(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;velocityS30_value;"
            "velocityS30_title\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;300;Some title\n",
            encoding="utf-8")

        with pytest.raises(SiteXMLImportError):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                delim=";")

    def test_csv_to_sera_site_preserves_zero_uncertainty(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;resonanceFrequency_value;"
            "resonanceFrequency_uncertainty\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;1.5;0.0\n",
            encoding="utf-8")

        sera_site_dict = csv_to_sera_site(
            site_owner_csv=site_owner_csv,
            site_description_csv=site_description_csv,
            analysis_csv=analysis_csv,
            delim=";")

        analysis = sera_site_dict["quakeml:test/site/001"].analysis[0]
        assert analysis.resonance_frequency.value.uncertainty == 0.0

    def test_csv_to_sera_site_raises_sitexml_import_error_for_missing_inputs(self):
        with pytest.raises(SiteXMLImportError):
            csv_to_sera_site(None, None)

    def test_csv_to_sera_site_raises_for_missing_required_owner_contact(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname\n"
            "TEST;Test Owner\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001\n",
            encoding="utf-8")

        with pytest.raises(SiteXMLImportError):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                delim=";")

    def test_csv_to_sera_site_raises_sitexml_io_error_for_missing_required_csv(self):
        with pytest.raises(SiteXMLIOError):
            csv_to_sera_site("missing_owner.csv", "missing_description.csv")

    def test_csv_to_sera_site_raises_for_invalid_optional_analysis_csv(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        with pytest.raises(SiteXMLIOError):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=tmp_path / "missing_analysis.csv",
                delim=";")

    def test_excel_to_sera_site_raises_for_invalid_optional_analysis_sheet(
            self, monkeypatch):
        workbook = "site_metadata.xlsx"
        monkeypatch.setattr(pd, "ExcelFile", lambda _: object())
        monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: {
            "siteOwner": pd.DataFrame([{
                "owner_codename": "TEST",
                "owner_fullname": "Test Owner",
                "person_firstname": "Name",
                "person_lastname": "Surname",
                "person_mbox": "someemail@domain.ab",
            }]),
            "siteDescription": pd.DataFrame([{
                "siteID": "quakeml:test/site/001",
                "siteDescriptionID": "quakeml:test/site_description/001",
                "latitude": 45.0,
                "longitude": 7.0,
            }]),
            "analysis": pd.DataFrame([{
                "siteID": "quakeml:test/site/001",
                "velocityS30_value": 300,
            }]),
        })

        with pytest.raises(SiteXMLImportError):
            excel_to_sera_site(workbook)
