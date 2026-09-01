#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for SiteXML CSV and Excel import helpers.

:author:
    Kiriaki Konstantinidou (kiriaki@itsak.gr), 2026
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import warnings

import pytest

try:
    import pandas as pd
except ImportError:
    HAS_PANDAS = False
else:
    HAS_PANDAS = True

import obspy
from obspy.io.sitexml.quality_index import (apply_quality_index_csv,
                                            apply_quality_index_dataframe,
                                            apply_quality_index_excel)
from obspy.io.sitexml.sitexml import sitexml_to_sitedict
from obspy.io.sitexml.scripts.csv2sitexml import main as csv2sitexml_main
from obspy.io.sitexml.scripts.excel2sitexml import (
    main as excel2sitexml_main)
from obspy.io.sitexml.tabular import (add_velocity_profiles, csv_to_sera_site,
                                      excel_to_sera_site, _read_year_cell)
from obspy.io.sitexml.util import SiteXMLImportError, SiteXMLIOError


@pytest.mark.skipif(not HAS_PANDAS, reason='pandas not installed')
class TestSiteXMLCSVImport():
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

    def _velocity_profile_dataframe(self, analysis_id=None):
        if analysis_id is None:
            analysis_id = "quakeml:domain.ab/analysis/001"
        return pd.DataFrame({
            "siteID": ["quakeml:domain.ab/site/001"] * 2,
            "analysisID": [analysis_id] * 2,
            "velocityProfileID": [
                "quakeml:domain.ab/velocity_profile/new"] * 2,
            "layerCount": [1, 2],
            "density_value": [None, None],
            "density_uncertainty": [None, None],
            "velocityP_value": [None, None],
            "velocityP_uncertainty": [None, None],
            "velocityS_value": [100.0, 250.0],
            "velocityS_uncertainty": [None, None],
            "layerTopDepth_value": [0.0, 5.0],
            "layerTopDepth_uncertainty": [None, None],
            "layerBottomDepth_value": [5.0, None],
            "layerBottomDepth_uncertainty": [None, None],
        })

    def _site_without_velocity_profiles(self, testdata):
        sera_site = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])["quakeml:domain.ab/site/001"]
        sera_site.get_analysis(
            "quakeml:domain.ab/analysis/001").velocity_profile_set = None
        return sera_site

    def test_add_velocity_profiles_detects_csv_and_updates_existing_site(
            self, testdata, tmp_path):
        sera_site = self._site_without_velocity_profiles(testdata)
        csv_path = tmp_path / "velocity_profiles.csv"
        self._velocity_profile_dataframe().to_csv(
            csv_path, sep=";", index=False)

        result = add_velocity_profiles(sera_site, csv_path)

        analysis = sera_site.get_analysis("quakeml:domain.ab/analysis/001")
        profiles = analysis.velocity_profile_set.velocity_profiles
        assert result is sera_site
        assert len(profiles) == 1
        assert profiles[0].resource_id == (
            "quakeml:domain.ab/velocity_profile/new")
        assert len(profiles[0].velocity_profile_data) == 2
        assert profiles[0].velocity_profile_data[0].velocityS.value == 100.0

    def test_add_velocity_profiles_detects_excel_and_updates_existing_site(
            self, testdata, tmp_path):
        pytest.importorskip("openpyxl")
        sera_site = self._site_without_velocity_profiles(testdata)
        excel_path = tmp_path / "velocity_profiles.xlsx"
        self._velocity_profile_dataframe().to_excel(
            excel_path, index=False)

        add_velocity_profiles(sera_site, excel_path)

        analysis = sera_site.get_analysis("quakeml:domain.ab/analysis/001")
        profiles = analysis.velocity_profile_set.velocity_profiles
        assert len(profiles) == 1
        assert profiles[0].velocity_profile_data[1].top_depth.value == 5.0

    def test_add_velocity_profiles_rejects_unknown_analysis(
            self, testdata, tmp_path):
        sera_site = self._site_without_velocity_profiles(testdata)
        csv_path = tmp_path / "velocity_profiles.csv"
        self._velocity_profile_dataframe(
            analysis_id="quakeml:domain.ab/analysis/missing").to_csv(
                csv_path, sep=";", index=False)

        with pytest.raises(SiteXMLImportError, match="unknown analysisID"):
            add_velocity_profiles(sera_site, csv_path)

    def test_csv_to_sera_site_imports_sites_analysis_and_velocity_profiles(
            self, testdata):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=testdata["site_owner.csv"],
            site_description_csv=testdata["site_description.csv"],
            analysis_csv=testdata["site_analysis.csv"],
            velocity_profiles_csv=testdata["velocity_profiles"],
            delim=";")

        assert set(sera_site_dict) == {
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
            "quakeml:domain.ab/site/003",
        }

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        assert site_001.site_owner.owner_codename == "SITEOWNER"
        assert site_001.site_owner.address_postal_code == "12345"
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
        assert analysis_001.velocity_profile_set is not None
        assert len(analysis_001.velocity_profile_set.velocity_profiles) == 2

        first_profile = analysis_001.velocity_profile_set.velocity_profiles[0]
        assert first_profile.resource_id == (
            "quakeml:domain.ab/velocity_profile/001")
        assert first_profile.layer_count == 8
        assert len(first_profile.velocity_profile_data) == 8
        vp_data = first_profile.velocity_profile_data[0]
        assert vp_data.velocityS.value == 118.08
        assert vp_data.velocityS.uncertainty == 2.0
        assert vp_data.top_depth.value == 0.0
        assert vp_data.bottom_depth.value == 0.19

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
        assert analysis_002.velocity_profile_set is not None
        assert len(analysis_002.velocity_profile_set.velocity_profiles) == 3
        assert site_002.site_description.h800.quality_index is None
        assert site_002.site_description.geological_unit.quality_index is None

        site_003 = sera_site_dict["quakeml:domain.ab/site/003"]
        assert site_003.site_description.resource_id == (
            "quakeml:domain.ab/site_description/003")
        assert site_003.site_description.station_code is None
        assert site_003.site_description.topographyA == "T1"
        assert site_003.site_description.topographyB == "Flat"
        assert site_003.site_description.morphology == "Plain"
        assert site_003.site_description.preferred_site_analysisID is None
        assert site_003.site_description.preferred_velocity_profileID is None
        assert site_003.analysis is None

    def test_csv_to_sera_site_imports_vs30_quality_indexes(self, testdata):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=testdata["site_owner.csv"],
            site_description_csv=testdata["site_description.csv"],
            analysis_csv=testdata["site_analysis.csv"],
            velocity_profiles_csv=testdata["velocity_profiles"],
            delim=";")

        analysis_001 = (
            sera_site_dict["quakeml:domain.ab/site/001"].analysis[0]
        )

        assert analysis_001.velocity_s30.method_combined_qindex == "1.2"
        assert analysis_001.velocity_s30.manual_qindex == "1.0"

    def test_csv_to_sera_site_applies_quality_index_sidecar(
            self, testdata):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=testdata["site_owner.csv"],
            site_description_csv=testdata["site_description.csv"],
            analysis_csv=testdata["site_analysis.csv"],
            velocity_profiles_csv=testdata["velocity_profiles"],
            quality_index_csv=testdata["quality_index.csv"],
            delim=";")

        site = sera_site_dict["quakeml:domain.ab/site/001"]
        q2 = site.calculate_quality_index2()
        q3 = site.calculate_quality_index3(
            f0_vs30=1, f0_bedrock_depth=0, vs30_geology=1)

        assert site.site_description.ec8.quality_index == 0.875
        assert q3 == pytest.approx(2 / 3)
        assert site.site_description.overall_quality_index == pytest.approx(
            (q2 + q3) / 2)

        site_002 = sera_site_dict["quakeml:domain.ab/site/002"]
        assert site_002.site_description.ec8.quality_index == 0.875
        assert site_002.site_description.bedrock_depth.quality_index == 0.375
        assert site_002.site_description.overall_quality_index is not None

    def test_quality_index_sidecar_recalculates_existing_qindex1(
            self, testdata):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=testdata["site_owner.csv"],
            site_description_csv=testdata["site_description.csv"],
            analysis_csv=testdata["site_analysis.csv"],
            velocity_profiles_csv=testdata["velocity_profiles"],
            quality_index_csv=testdata["quality_index.csv"],
            delim=";")

        site = sera_site_dict["quakeml:domain.ab/site/001"]

        assert site.site_description.ec8.quality_index == 0.875
        assert site.site_description.bedrock_depth.quality_index == 0.25

    def test_apply_quality_index_csv_updates_existing_sitexml_dict(
            self, testdata):
        sera_site_dict = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = apply_quality_index_csv(
                sera_site_dict,
                testdata["quality_index.csv"],
                delim=";")

        site = sera_site_dict["quakeml:domain.ab/site/001"]
        q2 = site.calculate_quality_index2()
        q3 = site.calculate_quality_index3(
            f0_vs30=1, f0_bedrock_depth=0, vs30_geology=1)

        assert result is sera_site_dict
        assert site.site_description.ec8.quality_index == 0.875
        assert site.site_description.overall_quality_index == pytest.approx(
            (q2 + q3) / 2)
        assert any("unknown siteID quakeml:domain.ab/site/002" in
                   str(w.message) for w in caught)

    def test_apply_quality_index_dataframe_updates_existing_sitexml_dict(
            self, testdata):
        sera_site_dict = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])
        df_quality_index = pd.read_csv(testdata["quality_index.csv"], sep=";")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = apply_quality_index_dataframe(
                sera_site_dict, df_quality_index)

        site = sera_site_dict["quakeml:domain.ab/site/001"]
        q2 = site.calculate_quality_index2()
        q3 = site.calculate_quality_index3(
            f0_vs30=1, f0_bedrock_depth=0, vs30_geology=1)

        assert result is sera_site_dict
        assert site.site_description.ec8.quality_index == 0.875
        assert site.site_description.overall_quality_index == pytest.approx(
            (q2 + q3) / 2)
        assert any("unknown siteID quakeml:domain.ab/site/002" in
                   str(w.message) for w in caught)

    def test_csv_to_sera_site_rejects_invalid_q3_sidecar_value(
            self, testdata, tmp_path):
        quality_index_csv = tmp_path / "quality_index.csv"
        quality_index_csv.write_text(
            "siteID;f0_vs30\n"
            "quakeml:domain.ab/site/001;0.5\n",
            encoding="utf-8")

        with pytest.raises(SiteXMLImportError, match="must be 0 or 1"):
            csv_to_sera_site(
                site_owner_csv=testdata["site_owner.csv"],
                site_description_csv=testdata["site_description.csv"],
                analysis_csv=testdata["site_analysis.csv"],
                velocity_profiles_csv=testdata["velocity_profiles"],
                quality_index_csv=quality_index_csv,
                delim=";")

    def test_csv_to_sera_site_imports_full_reference_metadata(self, testdata):
        sera_site_dict = csv_to_sera_site(
            site_owner_csv=testdata["site_owner.csv"],
            site_description_csv=testdata["site_description.csv"],
            analysis_csv=testdata["site_analysis.csv"],
            velocity_profiles_csv=testdata["velocity_profiles"],
            delim=";")

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        analysis_001 = site_001.analysis[0]

        self._assert_full_reference_metadata(site_001.site_description.ec8)
        self._assert_full_reference_metadata(analysis_001.resonance_frequency)
        self._assert_full_reference_metadata(
            analysis_001.velocity_profile_set)

    def test_csv_to_sera_site_preserves_reference_only_velocity_profile_set(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;"
            "velocityProfileSet_qualityIndex;"
            "velocityProfileSet_title;velocityProfileSet_firstAuthor;"
            "velocityProfileSet_year\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;0.7;Velocity profile study;"
            "Author A.;2026\n",
            encoding="utf-8")

        sera_site_dict = csv_to_sera_site(
            site_owner_csv=site_owner_csv,
            site_description_csv=site_description_csv,
            analysis_csv=analysis_csv,
            delim=";")

        velocity_profile_set = (
            sera_site_dict["quakeml:test/site/001"]
            .analysis[0].velocity_profile_set)
        assert velocity_profile_set is not None
        assert velocity_profile_set.velocity_profiles == []
        assert velocity_profile_set.quality_index == 0.7
        assert velocity_profile_set.literature_source.title == (
            "Velocity profile study")

    def test_csv_to_sera_site_drops_quality_only_velocity_profile_set(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;"
            "velocityProfileSet_qualityIndex\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;0.7\n",
            encoding="utf-8")

        sera_site_dict = csv_to_sera_site(
            site_owner_csv=site_owner_csv,
            site_description_csv=site_description_csv,
            analysis_csv=analysis_csv,
            delim=";")

        assert (
            sera_site_dict["quakeml:test/site/001"]
            .analysis[0].velocity_profile_set) is None

    def test_csv2sitexml_main_writes_sitexml_files(
            self, testdata, tmp_path):
        output_folder = tmp_path / "sitexml"

        result = csv2sitexml_main([
            "-o", str(testdata["site_owner.csv"]),
            "-d", str(testdata["site_description.csv"]),
            "-a", str(testdata["site_analysis.csv"]),
            "-p", str(testdata["velocity_profiles"]),
            "--output-folder", str(output_folder),
        ])

        date_text = obspy.UTCDateTime().strftime("%d-%m-%Y")
        assert result == 0
        assert sorted(path.name for path in output_folder.glob("*.xml")) == [
            "Site_XX.ABCD_%s.xml" % date_text,
            "Site_YY.WXYZ_%s.xml" % date_text,
            "Site_domain.ab.003_%s.xml" % date_text,
        ]

    def test_csv2sitexml_main_ignores_preferred_ids_without_analysis(
            self, testdata, tmp_path):
        output_folder = tmp_path / "sitexml"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = csv2sitexml_main([
                "-o", str(testdata["site_owner.csv"]),
                "-d", str(testdata["site_description.csv"]),
                "--output-folder", str(output_folder),
            ])

        date_text = obspy.UTCDateTime().strftime("%d-%m-%Y")
        site_dict = sitexml_to_sitedict(
            output_folder / ("Site_XX.ABCD_%s.xml" % date_text))
        site = site_dict["quakeml:domain.ab/site/001"]

        assert result == 0
        assert site.analysis is None
        assert site.site_description.preferred_site_analysisID is None
        assert site.site_description.preferred_velocity_profileID is None
        assert any("Ignoring preferredSiteAnalysisID" in str(w.message)
                   for w in caught)
        assert any("Ignoring preferredVelocityProfileID" in str(w.message)
                   for w in caught)

    def test_csv2sitexml_main_ignores_preferred_velocity_without_profiles(
            self, testdata, tmp_path):
        output_folder = tmp_path / "sitexml"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = csv2sitexml_main([
                "-o", str(testdata["site_owner.csv"]),
                "-d", str(testdata["site_description.csv"]),
                "-a", str(testdata["site_analysis.csv"]),
                "--output-folder", str(output_folder),
            ])

        date_text = obspy.UTCDateTime().strftime("%d-%m-%Y")
        site_dict = sitexml_to_sitedict(
            output_folder / ("Site_XX.ABCD_%s.xml" % date_text))
        site = site_dict["quakeml:domain.ab/site/001"]

        assert result == 0
        assert site.site_description.preferred_site_analysisID == (
            "quakeml:domain.ab/analysis/001")
        assert site.site_description.preferred_velocity_profileID is None
        assert any("velocity-profile metadata was not provided" in
                   str(w.message) for w in caught)

    def test_csv2sitexml_main_does_not_write_overall_qindex_without_qindex1(
            self, testdata, tmp_path):
        output_folder = tmp_path / "sitexml"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = csv2sitexml_main([
                "-o", str(testdata["site_owner.csv"]),
                "-d", str(testdata["minimal_site_description.csv"]),
                "-q", str(testdata["quality_index.csv"]),
                "--output-folder", str(output_folder),
            ])

        date_text = obspy.UTCDateTime().strftime("%d-%m-%Y")
        site_dict = sitexml_to_sitedict(
            output_folder / ("Site_XX.ABCD_%s.xml" % date_text))
        site = site_dict["quakeml:domain.ab/site/001"]

        assert result == 0
        assert site.site_description.overall_quality_index is None
        assert any("Analysis metadata not provided." in str(w.message)
                   for w in caught)

    def test_excel_to_sera_site_imports_sites_analysis_and_velocity_profiles(
            self, testdata):
        pytest.importorskip("openpyxl")

        sera_site_dict = excel_to_sera_site(
            path_or_file_object=testdata["full_site.xlsx"],
            velocity_profiles=testdata["velocity_profiles.xlsx"])

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
        assert analysis_001.velocity_profile_set is not None
        assert len(analysis_001.velocity_profile_set.velocity_profiles) == 2

        first_profile = analysis_001.velocity_profile_set.velocity_profiles[0]
        assert first_profile.resource_id == (
            "quakeml:domain.ab/velocity_profile/001")
        assert first_profile.layer_count == 8
        assert len(first_profile.velocity_profile_data) == 8
        vp_data = first_profile.velocity_profile_data[0]
        assert vp_data.velocityS.value == 118.08
        assert vp_data.velocityS.uncertainty == 2.0
        assert vp_data.top_depth.value == 0.0
        assert vp_data.bottom_depth.value == 0.19

        site_002 = sera_site_dict["quakeml:domain.ab/site/002"]
        assert site_002.site_description.resource_id == (
            "quakeml:domain.ab/site_description/002")
        assert len(site_002.analysis) == 1

        analysis_002 = site_002.analysis[0]
        assert analysis_002.resource_id == "quakeml:domain.ab/analysis/004"
        assert analysis_002.resonance_frequency.value.value == 0.3
        assert analysis_002.velocity_s30.value.value == 497.0
        assert analysis_002.velocity_s30.methods == ["S-REFL"]
        assert analysis_002.velocity_profile_set is not None
        assert len(analysis_002.velocity_profile_set.velocity_profiles) == 3

    def test_excel_to_sera_site_imports_all_reference_metadata(self, testdata):
        pytest.importorskip("openpyxl")

        sera_site_dict = excel_to_sera_site(
            path_or_file_object=testdata["full_site.xlsx"],
            velocity_profiles=testdata["velocity_profiles.xlsx"])

        site_001 = sera_site_dict["quakeml:domain.ab/site/001"]
        analysis_001 = site_001.analysis[0]

        self._assert_full_reference_metadata(site_001.site_description.ec8)
        self._assert_full_reference_metadata(analysis_001.resonance_frequency)
        self._assert_full_reference_metadata(
            analysis_001.velocity_profile_set)

    def test_excel2sitexml_main_writes_sitexml_files(
            self, testdata, tmp_path):
        pytest.importorskip("openpyxl")
        output_folder = tmp_path / "sitexml"

        result = excel2sitexml_main([
            str(testdata["full_site.xlsx"]),
            "-p", str(testdata["velocity_profiles.xlsx"]),
            "--output-folder", str(output_folder),
        ])

        date_text = obspy.UTCDateTime().strftime("%d-%m-%Y")
        assert result == 0
        assert sorted(path.name for path in output_folder.glob("*.xml")) == [
            "Site_XX.ABCD_%s.xml" % date_text,
            "Site_YY.WXYZ_%s.xml" % date_text,
        ]

    def test_excel_to_sera_site_applies_quality_index_sheet(self, testdata):
        pytest.importorskip("openpyxl")

        sera_site_dict = excel_to_sera_site(
            path_or_file_object=testdata["full_site.xlsx"],
            velocity_profiles=testdata["velocity_profiles.xlsx"])

        site = sera_site_dict["quakeml:domain.ab/site/001"]
        q2 = site.calculate_quality_index2()
        q3 = site.calculate_quality_index3(
            f0_vs30=1, f0_bedrock_depth=0, vs30_geology=1)
        assert site.site_description.ec8.quality_index == 0.875
        assert q3 == pytest.approx(2 / 3)
        assert site.site_description.overall_quality_index == pytest.approx(
            (q2 + q3) / 2)

        site_002 = sera_site_dict["quakeml:domain.ab/site/002"]
        assert site_002.site_description.ec8.quality_index == 0.875
        assert site_002.site_description.bedrock_depth.quality_index == 0.375
        assert site_002.site_description.overall_quality_index is not None

    def test_apply_quality_index_excel_updates_existing_sitexml_dict(
            self, testdata):
        pytest.importorskip("openpyxl")
        sera_site_dict = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = apply_quality_index_excel(
                sera_site_dict,
                testdata["full_site.xlsx"])

        site = sera_site_dict["quakeml:domain.ab/site/001"]
        q2 = site.calculate_quality_index2()
        q3 = site.calculate_quality_index3(
            f0_vs30=1, f0_bedrock_depth=0, vs30_geology=1)

        assert result is sera_site_dict
        assert site.site_description.ec8.quality_index == 0.875
        assert site.site_description.overall_quality_index == pytest.approx(
            (q2 + q3) / 2)
        assert any("unknown siteID quakeml:domain.ab/site/002" in
                   str(w.message) for w in caught)

    def test_excel_to_sera_site_warns_when_analysis_sheet_is_missing(
            self, testdata, tmp_path):
        pytest.importorskip("openpyxl")
        excel_path = tmp_path / "site_without_analysis.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            pd.read_csv(testdata["site_owner.csv"], sep=";").to_excel(
                writer, sheet_name="siteOwner", index=False)
            pd.read_csv(
                testdata["site_description.csv"], sep=";").to_excel(
                    writer, sheet_name="siteDescription", index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sera_site_dict = excel_to_sera_site(
                path_or_file_object=excel_path,
                velocity_profiles=testdata["velocity_profiles.xlsx"])

        assert set(sera_site_dict) == {
            "quakeml:domain.ab/site/001",
            "quakeml:domain.ab/site/002",
            "quakeml:domain.ab/site/003",
        }
        site = sera_site_dict["quakeml:domain.ab/site/001"]
        assert site.site_description.preferred_site_analysisID is None
        assert site.site_description.preferred_velocity_profileID is None
        assert any("Analysis metadata not provided." in str(w.message)
                   for w in caught)
        assert any("Ignoring preferredSiteAnalysisID" in str(w.message)
                   for w in caught)
        assert any("Ignoring preferredVelocityProfileID" in str(w.message)
                   for w in caught)

    def test_excel_to_sera_site_raises_for_missing_required_owner_sheet(
            self, testdata):
        pytest.importorskip("openpyxl")

        with pytest.raises(SiteXMLImportError):
            excel_to_sera_site(testdata["site_without_owner.xlsx"])

    def test_excel_to_sera_site_raises_for_missing_site_description_sheet(
            self, testdata):
        pytest.importorskip("openpyxl")

        with pytest.raises(SiteXMLImportError):
            excel_to_sera_site(
                testdata["site_without_site_description.xlsx"])

    def test_csv_to_sera_site_raises_for_invalid_site_description_rows(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude;station\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0;XX.AAA\n"
            ";quakeml:test/site_description/002;46.0;8.0;XX.BBB\n"
            "quakeml:test/site/003;;47.0;9.0;XX.CCC\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001\n",
            encoding="utf-8")

        with pytest.raises(
                SiteXMLImportError,
                match="Site description metadata row"):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                delim=";")

    def test_csv_to_sera_site_raises_for_invalid_optional_analysis_rows(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;velocityS30_value\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;300\n"
            "quakeml:test/site/001;;quakeml:test/site_description/001;250\n",
            encoding="utf-8")

        with pytest.raises(
                SiteXMLImportError,
                match="Analysis metadata row"):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                delim=";")

    def test_csv_to_sera_site_allows_missing_optional_columns(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;"
            "velocityProfileSet_qualityIndex\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;1\n",
            encoding="utf-8")

        velocity_profiles_csv = tmp_path / "velocity_profiles.csv"
        velocity_profiles_csv.write_text(
            "siteID;analysisID;velocityProfileID;velocityS_value;"
            "layerTopDepth_value\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/velocity_profile/001;120;0\n",
            encoding="utf-8")

        quality_index_csv = tmp_path / "quality_index.csv"
        quality_index_csv.write_text(
            "siteID\n"
            "quakeml:test/site/001\n",
            encoding="utf-8")

        sera_site_dict = csv_to_sera_site(
            site_owner_csv=site_owner_csv,
            site_description_csv=site_description_csv,
            analysis_csv=analysis_csv,
            velocity_profiles_csv=velocity_profiles_csv,
            quality_index_csv=quality_index_csv,
            delim=";")

        site = sera_site_dict["quakeml:test/site/001"]
        profile = site.analysis[0].velocity_profile_set.velocity_profiles[0]
        layer = profile.velocity_profile_data[0]
        assert layer.velocityS.value == 120
        assert layer.velocityS.uncertainty is None
        assert layer.bottom_depth is None

    def test_csv_to_sera_site_raises_for_missing_velocityS_column(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;"
            "velocityProfileSet_qualityIndex\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;1\n",
            encoding="utf-8")

        velocity_profiles_csv = tmp_path / "velocity_profiles.csv"
        velocity_profiles_csv.write_text(
            "siteID;analysisID;velocityProfileID;layerTopDepth_value\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/velocity_profile/001;0\n",
            encoding="utf-8")

        with pytest.raises(
                SiteXMLImportError,
                match="Velocity-profile metadata.*velocityS_value"):
            csv_to_sera_site(
                site_owner_csv=site_owner_csv,
                site_description_csv=site_description_csv,
                analysis_csv=analysis_csv,
                velocity_profiles_csv=velocity_profiles_csv,
                delim=";")

    def test_apply_quality_index_dataframe_requires_site_id_column(
            self, testdata):
        sera_site_dict = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])
        df_quality_index = pd.DataFrame([{"f0_vs30": 1}])

        with pytest.raises(
                SiteXMLImportError,
                match="Quality-index metadata.*siteID"):
            apply_quality_index_dataframe(sera_site_dict, df_quality_index)

    def test_apply_quality_index_dataframe_skips_missing_site_id_value(
            self, testdata):
        sera_site_dict = sitexml_to_sitedict(
            testdata["full_sitexml.xml"])
        df_quality_index = pd.DataFrame([{
            "siteID": "",
            "siteClassEC8_method": "documented",
            "siteClassEC8_evaluation": "direct",
            "siteClassEC8_reliability": "yes",
            "siteClassEC8_report": "yes",
        }])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = apply_quality_index_dataframe(
                sera_site_dict, df_quality_index)

        assert result is sera_site_dict
        assert any("missing siteID value" in str(w.message) for w in caught)

    def test_csv_to_sera_site_raises_for_incomplete_literature_source(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
            encoding="utf-8")

        analysis_csv = tmp_path / "site_analysis.csv"
        analysis_csv.write_text(
            "siteID;analysisID;siteDescriptionID;velocityS30_value;"
            "velocityS30_title\n"
            "quakeml:test/site/001;quakeml:test/analysis/001;"
            "quakeml:test/site_description/001;300;Some title\n",
            encoding="utf-8")

        with pytest.raises(
                SiteXMLImportError,
                match="requires both title and firstAuthor"):
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
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
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

    def test_csv_to_sera_site_raises_import_error_for_missing_inputs(self):
        with pytest.raises(SiteXMLImportError):
            csv_to_sera_site(None, None)

    def test_csv_to_sera_site_raises_for_missing_required_owner_contact(
            self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname\n"
            "TEST;Test Owner\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
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

    def test_csv_to_sera_site_raises_io_error_for_missing_required_csv(self):
        with pytest.raises(SiteXMLIOError):
            csv_to_sera_site("missing_owner.csv", "missing_description.csv")

    def test_csv_to_sera_site_raises_for_invalid_analysis_csv(self, tmp_path):
        site_owner_csv = tmp_path / "site_owner.csv"
        site_owner_csv.write_text(
            "owner_codename;owner_fullname;person_firstname;"
            "person_lastname;person_mbox\n"
            "TEST;Test Owner;Name;Surname;someemail@domain.ab\n",
            encoding="utf-8")

        site_description_csv = tmp_path / "site_description.csv"
        site_description_csv.write_text(
            "siteID;siteDescriptionID;latitude;longitude\n"
            "quakeml:test/site/001;"
            "quakeml:test/site_description/001;45.0;7.0\n",
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
