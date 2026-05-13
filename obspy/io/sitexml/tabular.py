# -*- coding: utf-8 -*-
"""
Functions dealing with tabular SiteXML metadata imports from CSV and Excel
files.

:copyright:
    ORFEUS, 2026
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from pathlib import Path
import os
import warnings

import pandas as pd
from collections import defaultdict

import obspy
from obspy.core.inventory.util import ExternalReference
from .core import (SERASite, SiteDescription, SERASiteOwner, Analysis,
                   EC8, H800, BedrockDepth, GeologicalUnit, 
                   ResonanceFrequency, VelocityS30, 
                   VelocityProfile, VelocityProfileData, VelocityProfileSet, 
                   LiteratureSource, ValueWithUncertainty)
from .quality_index import apply_quality_index_dataframe
from .util import (SiteXMLIOError, SiteXMLImportError,
                   SiteXMLValidationError)


def _csv_to_dataframe(path_or_file_object, label, delim=';'):
    """
    Read CSV tabular metadata as a dataframe with SiteXML exceptions.

    :rtype: :class:`pandas.DataFrame`
    """
    try:
        return pd.read_csv(path_or_file_object, sep=delim)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access {label}: {path_or_file_object}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read {label}: {path_or_file_object}"
        ) from e


def _excel_to_dataframe(path_or_file_object, label, sheet_name=0,
                        converters=None, missing_sheet_message=None):
    """
    Read Excel tabular metadata as dataframe(s) with SiteXML exceptions.

    :rtype: :class:`pandas.DataFrame` or dict
    """
    try:
        xls = pd.ExcelFile(path_or_file_object)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access {label}: {path_or_file_object}"
        ) from e

    try:
        return pd.read_excel(
            xls, sheet_name=sheet_name, converters=converters)
    except ValueError as e:
        if missing_sheet_message is not None:
            raise SiteXMLImportError(missing_sheet_message) from e
        raise SiteXMLImportError(
            f"Could not read {label}: {path_or_file_object}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read {label}: {path_or_file_object}"
        ) from e


def _read_dataframe_metadata(reader, *args, context, **kwargs):
    """
    Convert dataframe metadata to SiteXML objects with phase context.
    """
    try:
        return reader(*args, **kwargs)
    except SiteXMLImportError:
        raise
    except Exception as e:
        raise SiteXMLImportError(f"Could not build {context}.") from e


def _require_dataframe_columns(df, columns, context):
    """
    Raise if required dataframe columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SiteXMLImportError(
            f"{context} is missing required column(s): "
            + ", ".join(missing)
        )


def _require_row_values(row, columns, context):
    """
    Raise if required row values are missing or empty.
    """
    missing = [
        column for column in columns
        if column not in row or _empty_value(row[column])
    ]
    if missing:
        raise SiteXMLImportError(
            f"{context} is missing required value(s): "
            + ", ".join(missing)
        )


def csv_to_sera_site(site_owner_csv,
                     site_description_csv, 
                     analysis_csv=None, 
                     velocity_profiles_csv=None, 
                     quality_index_csv=None,
                     delim=';'):
    """
    Function to import SiteXML metadata from CSV files.

    :type site_owner_csv: str, pathlib.Path, or file-like object, required
    :param site_owner_csv: One line csv file with site owner metadata.
    :type site_description_csv: str, pathlib.Path, or file-like object, required
    :param site_description_csv: CSV file with site description metadata. One
        line per station/location.
    :type analysis_csv: str, pathlib.Path, or file-like object, optional
    :param analysis_csv: CSV file with analysis metadata. One line per
        analysisID. If omitted, preferred analysis and velocity-profile IDs
        read from the site-description CSV are ignored with a warning.
    :type velocity_profiles_csv: str, pathlib.Path, or file-like object, optional
    :param velocity_profiles_csv: CSV file or path to a folder with velocity
        profile metadata. The folder can contain any number of CSV files. If
        omitted, preferred velocity-profile IDs read from the site-description
        CSV are ignored with a warning.
    :type quality_index_csv: str, pathlib.Path, or file-like object, optional
    :param quality_index_csv: CSV with extra quality-index calculation
        inputs. Values are used immediately to calculate SiteXML quality
        indexes and are not stored.
    :type delim: str, optional
    :param delim: CSV file delimiter. Default is semicolon-delimited.
    :rtype: dict of :class:`~obspy.io.sitexml.core.SERASite`
    :return: Returns a dictionary of SERASite objects. Dictionary keys are the
        unique SiteIDs.

    Example

    >>> from obspy.io.sitexml.tabular import csv_to_sera_site
    >>> sera_site_dict = csv_to_sera_site(  # doctest: +SKIP
    ...     "site_owner.csv", "site_description.csv", "analysis.csv",
    ...     "velocity_profiles.csv")
    """
    # This is probably not needed as these two arguments are mandatory.
    #
    if site_owner_csv is None or site_description_csv is None:
        raise SiteXMLImportError(
            "The site owner and site description metadata are mandatory."
        )
    
    df_site_owner = _csv_to_dataframe(
        site_owner_csv, "site owner CSV metadata", delim=delim)
    df_site_description = _csv_to_dataframe(
        site_description_csv, "site description CSV metadata", delim=delim)
    
    # Try to read the analysis metadata if a csv is provided.
    #
    if analysis_csv:
        df_analysis = _csv_to_dataframe(
            analysis_csv, "analysis CSV metadata", delim=delim)
        exists_analysis = True
    else:
        warnings.warn("Analysis metadata not provided.", UserWarning)
        exists_analysis = False

    # Read the velocity profiles and store them
    # in a dictionary of dataframes with key the siteID 
    #
    df_vp_dict = _import_velocity_profiles(
        velocity_profiles_csv, kind="CSV", delim=delim)
    
    site_owner = _read_dataframe_metadata(
        _read_site_owner, df_site_owner, context="site owner metadata")
    site_description_dict = _read_dataframe_metadata(
        _read_site_description, df_site_description,
        context="site description metadata")

    if exists_analysis:
        analysis_dict = _read_dataframe_metadata(
            _read_analysis, df_analysis, df_vp_dict,
            skip_invalid_rows=False, context="analysis metadata")
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(
            site_owner = site_owner,
            site_description = site_description_dict[siteID],
            created = None,
            resource_id = siteID)
        if exists_analysis and siteID in analysis_dict:
            sera_site_dict[siteID].analysis = analysis_dict[siteID]

    _clear_preferred_ids_without_target_metadata(
        sera_site_dict,
        has_analysis_metadata=exists_analysis,
        has_velocity_profile_metadata=df_vp_dict is not None)

    if quality_index_csv:
        df_quality_index = _csv_to_dataframe(
            quality_index_csv, "quality-index CSV metadata", delim=delim)
        apply_quality_index_dataframe(sera_site_dict, df_quality_index)

    return sera_site_dict

def excel_to_sera_site(path_or_file_object, velocity_profiles=None):
    """
    Function to import SiteXML metadata from Excel files.

    The Excel file should contain the following sheets:

    * ``siteOwner``: site ownership metadata. Mandatory.
    * ``siteDescription``: site description metadata. Mandatory.
    * ``analysis``: analysis metadata. Optional.
    * ``qualityIndex``: quality indexes calculation parameters. Optional

    If the optional analysis sheet is omitted, preferred analysis and
    velocity-profile IDs read from ``siteDescription`` are ignored with a
    warning. If velocity-profile metadata is omitted, preferred
    velocity-profile IDs read from ``siteDescription`` are ignored with a
    warning.

    :type path_or_file_object: str, pathlib.Path, or file-like object, required
    :param path_or_file_object: Excel file with site metadata.
    :type velocity_profiles: str, optional
    :param velocity_profiles: Excel file or path to a folder with velocity
        profile metadata.
    :rtype: dict of :class:`~obspy.io.sitexml.core.SERASite`
    :return: Returns a dictionary of SERASite objects. Dictionary keys are the
        unique SiteIDs.
    
    Example

    >>> from obspy.io.sitexml.tabular import excel_to_sera_site
    >>> sera_site_dict = excel_to_sera_site(  # doctest: +SKIP
    ...     "InputExcel.xlsx", velocity_profiles="vp.xlsx")
    """
    conv_dict = {
        'velocityS30_year': _read_year_cell,
        'velocityProfileSet_year': _read_year_cell,
        'resonanceFrequency_year': _read_year_cell,
        'siteClassEC8_year': _read_year_cell,
        'bedrockDepth_year': _read_year_cell,
        'h800_year': _read_year_cell,
        'geologicalUnit_year': _read_year_cell,
    }
    df_dict = _excel_to_dataframe(
        path_or_file_object, "Site metadata Excel file", sheet_name=None,
        converters=conv_dict)

    if not all(k in df_dict for k in ("siteOwner", "siteDescription")):
        raise SiteXMLImportError(
            "The site owner and site description sheets are mandatory."
        )
    
    # Read the velocity profiles and store them
    # in a dictionary of dataframes with key the siteID
    #
    df_vp_dict = _import_velocity_profiles(velocity_profiles, kind="Excel") \
        if velocity_profiles else None
    
    site_owner = _read_dataframe_metadata(
        _read_site_owner, df_dict['siteOwner'],
        context="site owner metadata")
    site_description_dict = _read_dataframe_metadata(
        _read_site_description, df_dict['siteDescription'],
        context="site description metadata")

    # Read the analysis metadata if the 'analysis' sheet exists.
    #
    if 'analysis' in df_dict:
        exists_analysis = True
        analysis_dict = _read_dataframe_metadata(
            _read_analysis, df_dict['analysis'], df_vp_dict,
            skip_invalid_rows=False, context="analysis metadata")
    else:
        warnings.warn("Analysis metadata not provided.", UserWarning)
        exists_analysis = False
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(
            site_owner = site_owner,
            site_description = site_description_dict[siteID],
            created = None,
            resource_id = siteID)
        if exists_analysis and siteID in analysis_dict:
            sera_site_dict[siteID].analysis = analysis_dict[siteID]

    _clear_preferred_ids_without_target_metadata(
        sera_site_dict,
        has_analysis_metadata=exists_analysis,
        has_velocity_profile_metadata=df_vp_dict is not None)

    if "qualityIndex" in df_dict:
        apply_quality_index_dataframe(sera_site_dict, df_dict["qualityIndex"])

    return sera_site_dict


def _clear_preferred_ids_without_target_metadata(
        sera_site_dict, has_analysis_metadata, has_velocity_profile_metadata):
    """
    Drop preferred IDs that point to metadata tables omitted from tabular input.
    """
    for site_id, sera_site in sera_site_dict.items():
        site_description = sera_site.site_description
        if not has_analysis_metadata:
            preferred_id = site_description.preferred_site_analysisID
            if preferred_id is not None:
                warnings.warn(
                    f"Site {site_id} provides preferredSiteAnalysisID "
                    f"{preferred_id}, but analysis metadata was not provided. "
                    "Ignoring preferredSiteAnalysisID for this import; the "
                    "generated SiteXML will omit that unresolved preference.",
                    UserWarning)
                site_description.preferred_site_analysisID = None

        if not has_analysis_metadata or not has_velocity_profile_metadata:
            preferred_id = site_description.preferred_velocity_profileID
            if preferred_id is not None:
                missing = (
                    "analysis metadata"
                    if not has_analysis_metadata
                    else "velocity-profile metadata")
                warnings.warn(
                    f"Site {site_id} provides preferredVelocityProfileID "
                    f"{preferred_id}, but {missing} was not provided. "
                    "Ignoring preferredVelocityProfileID for this import; the "
                    "generated SiteXML will omit that unresolved preference.",
                    UserWarning)
                site_description.preferred_velocity_profileID = None


def add_velocity_profiles(sera_sites, velocity_profiles, replace_existing=False,
                          delim=';'):
    """
    Add velocity profiles from CSV or Excel tabular metadata to existing sites.

    The velocity-profile input uses the same columns as ``csv_to_sera_site``
    and ``excel_to_sera_site`` sidecar velocity-profile inputs, including
    ``siteID``, ``analysisID``, and ``velocityProfileID``. File type is
    detected from the input path extension. Directory inputs must contain only
    CSV files or only Excel files.

    :type sera_sites: :class:`~obspy.io.sitexml.core.SERASite` or dict
    :param sera_sites: Existing SiteXML object or dictionary keyed by site ID.
    :type velocity_profiles: str or pathlib.Path, required
    :param velocity_profiles: CSV/Excel file or directory of CSV/Excel files.
    :type replace_existing: bool, optional
    :param replace_existing: Replace the target analysis velocity profiles
        instead of appending them. Existing ``VelocityProfileSet`` metadata is
        preserved.
    :type delim: str, optional
    :param delim: CSV file delimiter. Default is semicolon-delimited.
    :return: The original ``sera_sites`` object.

    Example:

    >>> from obspy.io.sitexml.sitexml import read_sitexml
    >>> from obspy.io.sitexml.tabular import add_velocity_profiles
    >>> site = read_sitexml("site.xml")  # doctest: +SKIP
    >>> add_velocity_profiles(  # doctest: +SKIP
    ...     site,
    ...     "velocity_profiles.csv",
    ...     replace_existing=True)
    """
    if isinstance(sera_sites, SERASite):
        sera_site_dict = {sera_sites.resource_id: sera_sites}
    else:
        sera_site_dict = sera_sites

    df_vp_dict = _import_velocity_profiles(
        velocity_profiles, delim=delim)
    if not df_vp_dict:
        return sera_sites

    for siteID, df_site in df_vp_dict.items():
        if siteID not in sera_site_dict:
            raise SiteXMLImportError(
                f"Velocity-profile input references unknown siteID {siteID}.")
        sera_site = sera_site_dict[siteID]

        for analysisID in df_site["analysisID"].dropna().unique():
            analysis = sera_site.get_analysis(analysisID)
            if analysis is None:
                raise SiteXMLImportError(
                    "Velocity-profile input references unknown analysisID "
                    f"{analysisID} for siteID {siteID}.")

            new_profiles = _read_velocity_profiles_for_analysis(
                df_site, analysis_id=analysisID)
            if not new_profiles:
                continue

            try:
                sera_site.add_velocity_profiles(
                    new_profiles,
                    analysisID=analysisID,
                    replace_existing=replace_existing)
            except SiteXMLValidationError as exc:
                raise SiteXMLImportError(
                    "Could not add velocity profiles from tabular input."
                ) from exc

    return sera_sites


def _read_site_description(df_site_description):
    """
    Return site-description objects keyed by site ID from tabular metadata.

    :rtype: dict of :class:`~obspy.io.sitexml.core.SiteDescription`
    :return: A dictionary of SiteDescription objects. Dictionary keys are the
        unique SiteIDs.
    """
    required_columns = ("siteID", "siteDescriptionID", "latitude", "longitude")
    _require_dataframe_columns(
        df_site_description, required_columns, "Site description metadata")

    site_description_dict = {}

    for index, row in df_site_description.iterrows():

        _require_row_values(
            row, required_columns,
            f"Site description metadata row {index}")

        siteID = _read_cell(row, "siteID")
        resource_id =  _read_cell(row, "siteDescriptionID")
        latitude = _read_cell(row, "latitude")
        longitude = _read_cell(row, "longitude")
        
        station_code = _read_cell(row, "station")
        
        site_description_obj = SiteDescription(resource_id=resource_id,
                                       station_code=station_code, 
                                       latitude=latitude, 
                                       longitude=longitude)
        
        site_description_obj.altitude = \
            _read_cell(row, "altitude")
        site_description_obj.min_distance_from_station = \
            _read_cell(row, "minDistanceFromStation")
        site_description_obj.max_distance_from_station = \
            _read_cell(row, "maxDistanceFromStation")
        site_description_obj.morphology = \
            _read_cell(row, "siteMorphology")
        site_description_obj.topographyA = \
            _read_cell(row, "siteTopography_schemaA")
        site_description_obj.topographyB = \
            _read_cell(row, "siteTopography_schemaB")
        site_description_obj.preferred_site_analysisID = \
            _read_cell(row, "preferredSiteAnalysisID")
        site_description_obj.preferred_velocity_profileID = \
            _read_cell(row, "preferredVelocityProfileID")
        site_description_obj.overall_quality_index = \
            _read_cell(row, "overallQindex")
        
        site_description_obj.ec8 = \
            _read_site_indicator(row, EC8, 'siteClassEC8')
        site_description_obj.bedrock_depth = \
            _read_site_indicator(row, BedrockDepth, 'bedrockDepth')
        site_description_obj.h800 = \
            _read_site_indicator(row, H800, 'h800')
        site_description_obj.geological_unit = \
            _read_site_indicator(row, GeologicalUnit, 'geologicalUnit')
        
        site_description_dict[siteID] = site_description_obj

    return site_description_dict

def _read_analysis(df_analysis, df_vp_dict=None, skip_invalid_rows=True):
    """
    Return a dictionary of Analysis objects for all sites.

    Dictionary key is the siteID.
    Analysis rows can be skipped only when ``skip_invalid_rows`` is true.
    The public CSV/Excel importers pass ``False`` so malformed analysis rows
    stop import.

    :type df_analysis: :class:`pandas.DataFrame`, required
    :param df_analysis: Dataframe with analysis metadata for all sites
    :type df_vp_dict: dict of :class:`pandas.DataFrame`, optional
    :param df_vp_dict: Dictionary of :class:`pandas.DataFrame` with velocity
            profile metadata for all sites. Dictionary key is the siteID.
    :rtype: dict of :class:`~obspy.io.sitexml.core.Analysis`
    :return: A dictionary of Analysis objects. Dictionary keys are the unique
        SiteIDs.
    """
    required_columns = ("siteID", "analysisID", "siteDescriptionID")
    _require_dataframe_columns(
        df_analysis, required_columns, "Analysis metadata")

    analysis_dict = defaultdict(list)

    for index, row in df_analysis.iterrows():

        # TODOs What if they don't provide the IDs in the csv file??
        #
        siteID = _read_cell(row, "siteID")
        analysisID = _read_cell(row, "analysisID")
        site_descriptionID = _read_cell(row, "siteDescriptionID")
        
        if siteID and analysisID and site_descriptionID:
            analysis_obj = Analysis(
                resource_id = analysisID,
                site_descriptionID = site_descriptionID)
                
            # Go on reading the site characterization indicators
            analysis_obj.resonance_frequency = _read_site_indicator(
                row, ResonanceFrequency, 'resonanceFrequency')
            analysis_obj.velocity_s30 = _read_site_indicator(
                row, VelocityS30, 'velocityS30')
            analysis_obj.velocity_profile_set = _read_site_indicator(
                row, VelocityProfileSet, 'velocityProfileSet')
            
            analysis_obj.spt_logs_count = \
                _read_cell(row, "sptLogsCount")
            analysis_obj.cpt_logs_count = \
                _read_cell(row, "cptLogsCount")
            analysis_obj.borehole_logs_count = \
                _read_cell(row, "boreholeLogsCount")
           
            # Read Velocity Profiles of Analysis
            #
            if df_vp_dict and siteID in df_vp_dict \
                and analysis_obj.velocity_profile_set:
                analysis_obj.velocity_profile_set.velocity_profiles = \
                    _read_velocity_profiles_for_analysis(
                        df_vp_dict[siteID],
                        analysis_id=analysisID)
            if not _has_velocity_profile_set_content(
                    analysis_obj.velocity_profile_set):
                analysis_obj.velocity_profile_set = None
            
            # Add analysis object in analysis_dict using as key the siteID
            analysis_dict[siteID].append(analysis_obj)
                                       
        else:
            if skip_invalid_rows:
                warnings.warn("Missing siteID, analysisID or siteDescriptionID "
                              "value. Processing of analysis element will be "
                              "skipped.", UserWarning)
                continue
            raise SiteXMLImportError(
                f"Analysis metadata row {index} is missing required "
                "siteID, analysisID or siteDescriptionID values. "
                "Abording further processing."
            )
    
    return analysis_dict


def _has_velocity_profile_set_content(velocity_profile_set):
    """
    Return whether a velocity-profile set has profile data or references.

    :rtype: bool
    """
    if velocity_profile_set is None:
        return False
    return bool(
        velocity_profile_set.velocity_profiles or
        velocity_profile_set.literature_source or
        velocity_profile_set.external_references)


def _read_velocity_profiles_for_analysis(df_vp, analysis_id):
    """
    Return a list of VelocityProfile objects for a given analysisID.

    Velocity-profile rows are not skipped. A malformed layer would make the
    containing velocity profile ambiguous, so required columns and values are
    validated before this reader groups rows into profiles.

    :type df_vp: :class:`pandas.DataFrame`, required
    :param df_vp: Dataframe of velocity profiles for a single site
    :type analysis_id: str, required
    :param analysis_id: The analysis for which to read velocity profiles
    :rtype: list of :class:`~obspy.io.sitexml.core.VelocityProfile`
    """

    # 1. Filter the DataFrame to this analysisID
    df_analysis = df_vp[df_vp["analysisID"] == analysis_id]

    if df_analysis.empty:
        return None
        
    velocity_profiles = []

    # 2. Group by velocityProfileID inside this analysis
    for profile_id, df_profile in df_analysis.groupby("velocityProfileID"):
        vp = _read_velocity_profile(df_profile)
        velocity_profiles.append(vp)

    return velocity_profiles

def _read_velocity_profile(rows):
    """
    Build a VelocityProfile object from rows belonging to a single profile.

    :param rows: A group of dataframe rows
    :rtype: :class:`~obspy.io.sitexml.core.VelocityProfile`
    """
    if "layerCount" in rows.columns:
        rows = rows.sort_values("layerCount")
    layer_objects = []

    for idx, row in rows.iterrows():

        density = _read_value_with_uncertainty(row, "density")        
        velP = _read_value_with_uncertainty(row, "velocityP")
        velS = _read_value_with_uncertainty(row, "velocityS", required=True)
        
        top_depth = _read_value_with_uncertainty(
            row, "layerTopDepth", required=True)
        bottom_depth = _read_value_with_uncertainty(row, "layerBottomDepth")

        layer_obj = VelocityProfileData(
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            density=density,
            velocityP=velP,
            velocityS=velS
        )

        layer_objects.append(layer_obj)

    return VelocityProfile(
        resource_id=rows.iloc[0]["velocityProfileID"],
        velocity_profile_data=layer_objects
    )

def _read_site_indicator(df_row, cls, indicator):
    """
    Build one site indicator from a tabular row and indicator prefix.

    :rtype: :class:`~obspy.io.sitexml.core.SiteIndicator` or None
    """
    if indicator != "velocityProfileSet":
        value_column = indicator + '_value'
        if value_column not in df_row or _empty_value(df_row[value_column]):
            return None
        
        obj = cls(value=df_row[value_column])
        
        if _read_cell(df_row, indicator+'_uncertainty') is not None:
            obj.value.uncertainty = df_row[indicator+'_uncertainty']

        if _read_cell(df_row, indicator+'Method1'):
            obj.methods.append(df_row[indicator+'Method1'])
        if _read_cell(df_row, indicator+'Method2'):
            obj.methods.append(df_row[indicator+'Method2'])
        
        if indicator == "velocityS30":
            method_combined_qindex = _read_cell(
                df_row, 'velocityS30MethodCombIndex')
            manual_qindex = _read_cell(df_row, 'velocityS30ManualIndex')
            obj.method_combined_qindex = (
                None if method_combined_qindex is None \
                else str(method_combined_qindex)
            )
            obj.manual_qindex = (
                None if manual_qindex is None else str(manual_qindex)
            )
        
        if indicator == "geologicalUnit":
            obj.value = obj.value[0:255]
            obj.geological_map_scale = _read_cell(df_row, 'geologicalMapScale')
            obj.geological_unit_OGE = _read_cell(df_row, 'geologicalUnitOGE')
    else:
        obj = cls()

    obj.literature_source = _read_literature_source(df_row, indicator)
    obj.external_references = _read_external_references(df_row, indicator)
    obj.quality_index = _read_cell(df_row, indicator+'Qindex1')
    
    return obj

def _import_velocity_profiles(path, kind=None, delim=';'):
    """
    Read velocity-profile files and return dataframes grouped by site ID.

    If ``kind`` is ``"CSV"`` or ``"Excel"``, only that tabular format is
    accepted. If ``kind`` is omitted, the format is detected from file suffixes.

    :rtype: dict or None
    """
    if not path:
        return None

    importers = {
        "CSV": {
            "extensions": (".csv",),
            "read_file": lambda file_path: _read_velocity_profile_csv_file(
                file_path, delim=delim),
        },
        "Excel": {
            "extensions": (".xls", ".xlsx", ".xlsm", ".xlsb"),
            "read_file": _read_velocity_profile_excel_file,
        },
    }
    if kind is not None and kind not in importers:
        raise SiteXMLImportError(
            f"Unknown velocity-profile tabular input kind: {kind}")

    def _kind_for_suffix(suffix):
        for importer_kind, importer in importers.items():
            if suffix in importer["extensions"]:
                return importer_kind
        return None

    path_str = os.fspath(path)
    df = pd.DataFrame()

    if os.path.isdir(path_str):
        filenames = [
            filename for filename in os.listdir(path_str)
            if os.path.isfile(os.path.join(path_str, filename))
        ]
        detected_kinds = {
            _kind_for_suffix(Path(filename).suffix.lower())
            for filename in filenames
        }
        detected_kinds.discard(None)

        if kind is None:
            unsupported = [
                filename for filename in filenames
                if _kind_for_suffix(Path(filename).suffix.lower()) is None
            ]
            if unsupported:
                raise SiteXMLImportError(
                    "Velocity-profile directory contains unsupported file "
                    "types.")
            if len(detected_kinds) > 1:
                raise SiteXMLImportError(
                    "Velocity-profile directory mixes CSV and Excel files.")
            if not detected_kinds:
                return None
            kind_name = detected_kinds.pop()
        else:
            kind_name = kind

        importer = importers[kind_name]
        for filename in filenames:
            file_path = os.path.join(path_str, filename)
            if not filename.lower().endswith(importer["extensions"]):
                raise SiteXMLImportError(
                    "Velocity-profile input is not a "
                    f"{kind_name} file: {file_path}"
                )
            df = pd.concat(
                [df, importer["read_file"](file_path)], ignore_index=True)
    elif os.path.isfile(path_str):
        suffix = Path(path_str).suffix.lower()
        kind_name = kind or _kind_for_suffix(suffix)
        if kind_name is None:
            raise SiteXMLImportError(
                f"Velocity-profile input is not a CSV or Excel file: {path_str}")
        importer = importers[kind_name]
        if not path_str.lower().endswith(importer["extensions"]):
            raise SiteXMLImportError(
                f"Velocity-profile input is not a {kind_name} file: {path_str}"
            )
        df = importer["read_file"](path_str)
    else:
        raise SiteXMLIOError(f"Velocity-profile path does not exist: {path_str}")

    if not df.empty:
        _validate_velocity_profile_dataframe(df, path_str)
        return {site_id: group for site_id, group in df.groupby("siteID")}
    return None


def _validate_velocity_profile_dataframe(df, source):
    """
    Validate required velocity-profile columns and row values.
    """
    required_columns = (
        "siteID", "analysisID", "velocityProfileID",
        "velocityS_value", "layerTopDepth_value")
    _require_dataframe_columns(
        df, required_columns, f"Velocity-profile metadata in {source}")
    for index, row in df.iterrows():
        _require_row_values(
            row, required_columns,
            f"Velocity-profile metadata row {index} in {source}")


def _read_velocity_profile_csv_file(file_path, delim=';'):
    """
    Read one velocity-profile CSV file as a dataframe.

    :rtype: :class:`pandas.DataFrame`
    """
    return _csv_to_dataframe(
        file_path, "velocity-profile CSV file", delim=delim)


def _read_velocity_profile_excel_file(file_path):
    """
    Read all non-empty velocity-profile sheets from one Excel file.

    :rtype: :class:`pandas.DataFrame`
    """
    df_dict = _excel_to_dataframe(
        file_path, "velocity-profile Excel file", sheet_name=None)

    df = pd.DataFrame()
    for sheet_name, sheet_df in df_dict.items():
        if sheet_df is None or sheet_df.empty:
            continue
        if "siteID" not in sheet_df.columns:
            raise SiteXMLImportError(
                f"Missing required 'siteID' column in sheet {sheet_name} "
                f"of {file_path}."
            )
        df = pd.concat([df, sheet_df], ignore_index=True)
    return df

def _read_literature_source(df_row, indicator):
    """
    Return literature metadata for one indicator.

    :rtype: :class:`~obspy.io.sitexml.core.LiteratureSource` or None
    """

    title = _read_cell(df_row, 'title', indicator)
    first_author = _read_cell(df_row, 'firstAuthor', indicator)
    # Title and firstAuthor are required according to the schema.
    #
    if title and first_author:
        literature_source = LiteratureSource(title=title,
                                             first_author=first_author)
        literature_source.secondary_authors = _read_cell(
            df_row, 'secondaryAuthors', indicator)
        literature_source.year = _read_cell(df_row, 'year', indicator)
        literature_source.booktitle = _read_cell(df_row, 'booktitle', indicator)
        literature_source.language = _read_cell(df_row, 'language', indicator)
        literature_source.doi = _read_cell(df_row, 'doi', indicator)
        return literature_source
    elif title or first_author:
        raise SiteXMLImportError(
            f"{indicator} literature source requires both title and "
            "firstAuthor."
        )
    return None

def _read_external_references(df_row, indicator):
    """
    Return external references for one indicator.

    :rtype: list[:class:`~obspy.core.inventory.util.ExternalReference`] or None
    """

    uri = _read_cell(df_row, 'uri', indicator)
    description = _read_cell(df_row, 'description', indicator)
    if uri:
        return [ExternalReference(uri=uri, description=description)]
    return None

def _read_value_with_uncertainty(row, name, required=False):
    """
    Return a ValueWithUncertainty read from ``<name>_value`` columns.

    :rtype: :class:`~obspy.io.sitexml.core.ValueWithUncertainty` or None
    """

    value_column = name + "_value"
    uncertainty_column = name + "_uncertainty"
    value = _read_cell(row, value_column)
    if value is None:
        if required:
            raise SiteXMLImportError(
                f"Velocity-profile metadata is missing required value: "
                f"{value_column}")
        return None

    metric = ValueWithUncertainty(value)
    
    uncertainty = _read_cell(row, uncertainty_column)
    if uncertainty is not None:
        metric.uncertainty = uncertainty

    return metric

def _read_site_owner(df):
    """
    Return the required site-owner metadata from the first dataframe row.

    :rtype: :class:`~obspy.io.sitexml.core.SERASiteOwner`
    """
    if df.empty:
        raise SiteXMLImportError("Site owner metadata is mandatory.")

    row = next(df.iterrows())[1]
    required_attrs = (
        "owner_codename", "owner_fullname",
        "person_firstname", "person_lastname", "person_mbox")
    missing = [
        attr for attr in required_attrs
        if attr not in df.columns or _empty_value(row[attr])
    ]
    if missing:
        raise SiteXMLImportError(
            "Site owner metadata is missing required value(s): "
            + ", ".join(missing)
        )

    obj = SERASiteOwner(
        owner_codename=row["owner_codename"],
        owner_fullname=row["owner_fullname"],
        person_firstname=row["person_firstname"],
        person_lastname=row["person_lastname"],
        person_mbox=row["person_mbox"])

    for attr in vars(obj).keys():
        pub_attr = attr.strip('_')
        if pub_attr in df.columns and not _empty_value(row[pub_attr]):
            try:
                setattr(obj, pub_attr, row[pub_attr])
            except Exception as e:
                warnings.warn(
                    f"Could not set attribute '{pub_attr}' on SiteOwner: {e}",
                    UserWarning)
    return obj

def _read_cell(df_row, argument, indicator=None):
    """
    Return a non-empty cell value, optionally using an indicator prefix.

    :rtype: object or None
    """

    if indicator:
        if indicator + "_" + argument in df_row and \
            not _empty_value(df_row[indicator + "_" + argument]):
                    return df_row[indicator + "_" + argument]
    else:
        if argument in df_row and \
            not _empty_value(df_row[argument]):
                    return df_row[argument]
        
    return None

def _read_year_cell(value):
    """
    Normalize an Excel year cell to the schema's four-digit string form.

    :rtype: str or None
    """
    if _empty_value(value):
        return None
    if isinstance(value, float):
        if not value.is_integer():
            raise SiteXMLImportError(
                f"Year values must be four-digit years, got {value!r}."
            )
        value = int(value)
    value = str(value)
    if not value.isdigit() or len(value) != 4:
        raise SiteXMLImportError(
            f"Year values must be four-digit years, got {value!r}."
        )
    return value

def _empty_value(value):
    """
    Return whether a tabular cell should be treated as missing.

    :rtype: bool
    """
    if pd.isna(value):
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
    
    return False
