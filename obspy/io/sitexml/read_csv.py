# -*- coding: utf-8 -*-
"""
Functions dealing with import SiteXML metadata from excel files.

:copyright:
    ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from pathlib import Path
import os
import warnings
import re

import pandas as pd
from collections import defaultdict

import obspy
from obspy.core.inventory.util import ExternalReference
from .core import (SERASite, SiteDescription, SERASiteOwner, Analysis,
                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency,
                   VelocityS30, VelocityProfile, VelocityProfileData,
                   VelocityProfileSurvey, LiteratureSource, ValueWithUncertainty)
from .sitexml import write_sitexml
from .util import SiteXMLIOError, SiteXMLImportError

def sitedict_to_sitexml(sera_site_dict, output_folder="."):
    """
    Exports a dictionary of SERAsite objects to the respective SiteXML files. 
    The files are written to a folder given with argument "output_folder".
    The name of the SiteXML file is either:
    - the station_code if the metadata belong to a station site
    - The siteID otherwise

    :type sera_site_dict: dict of :class:`~obspy.io.sitexml.core.SERASite`
    :param sera_site_dict: Dictionary of SERAsite objects
    :type output_folder: str, optional
    :param output_folder: Output folder to write the SiteXMl files. 
                        If not provided writes to the current folder.
    :rtype: None
    """
    for sera_site in sera_site_dict.values():
        if sera_site.site_description.station_code:
            output_file = output_folder + "/" + sera_site.site_description.station_code + ".xml"
        else:
            filename = re.sub(r"[^A-Za-z0-9]+", "_", sera_site.resource_id).strip("_")
            output_file = output_folder + "/" + filename + ".xml"
        write_sitexml(sera_site, output_file, validate=True)

def csv_to_sera_site(site_owner_csv,
                     site_description_csv, 
                     analysis_csv=None, 
                     velocity_profiles_csv=None, 
                     delim='\t'):
    """
    Function to import SiteXML metadata from CSV files.
    The Excel file should contain the following sheets:
    - siteOwner: With site ownership metadata. Mandatory.
    - siteDescription: With site description metadata. Mandatory.
    - analysis: With analysis metadata. Optional

    Returns a dictionary of SERASite objects. Dictionary keys are the unique SiteIDs.

    :type site_owner_csv: File name or file like object, required
    :param site_owner_csv: One line csv file with site owner metadata.
    :type site_description_csv: File name or file like object, required
    :param site_description_csv: CSV file with site description metadata. 
            One line per station/location.
    :type analysis_csv: File name or file like object, optional
    :param analysis_csv: CSV file with analysis metadata. 
            One line per analysisID.
    :type velocity_profiles_csv: 
    :param velocity_profiles_csv: CSV file or path to a folder with velocity profile metadata.
            The folder can contain any number of CSV files. Optional.
    :type delim: str, optional
    :param delim: CSV file delimiter. Default tab delimeted.
    :rtype: dictionary of :class:`~obspy.io.sitexml.core.SERASite`

    Example

    >>> from obspy.io.sitexml.read_csv import csv_to_sera_site
    >>> sera_site_dict = csv_to_sera_site("site_owner.csv",
    ...                     "site_description.csv", "analysis.csv",
    ...                     "velocity_profiles_dir", ';')
    """
    # This is probably not needed as these two arguments are mandatory.
    #
    if site_owner_csv is None or site_description_csv is None:
        raise SiteXMLImportError(
            "The site owner and site description metadata are mandatory."
        )
    
    try:
        df_site_owner = pd.read_csv(site_owner_csv, sep=delim)
        df_site_description = pd.read_csv(site_description_csv, sep=delim)
    except OSError as e:
        raise SiteXMLIOError(
            "Could not access the required site owner or site description CSV "
            "metadata."
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            "Could not read the required site owner or site description CSV "
            "metadata."
        ) from e
    
    # Try to read the analysis metadata if a csv is provided.
    #
    if analysis_csv:
        try:
            df_analysis = pd.read_csv(analysis_csv, sep=delim)
            exists_analysis = True
        except OSError as e:
            raise SiteXMLIOError(
                f"Could not access analysis CSV metadata: {analysis_csv}"
            ) from e
        except Exception as e:
            raise SiteXMLImportError(
                f"Could not read analysis CSV metadata: {analysis_csv}"
            ) from e
    else:
        warnings.warn("Analysis metadata not provided.", UserWarning)
        exists_analysis = False

    # Read the velocity profiles and store them
    # in a dictionary of dataframes with key the siteID 
    #
    df_vp_dict = _csv_import_velocity_profiles(velocity_profiles_csv, delim=delim)
    
    #site_owner_dict = _read_sheet(df_site_owner, SERASiteOwner)
    site_owner = _read_site_owner(df_site_owner)
    site_description_dict = _read_site_description(df_site_description)

    if exists_analysis:
        try:
            analysis_dict = _read_analysis(
                df_analysis, df_vp_dict, skip_invalid_rows=False)
        except Exception as e:
            raise SiteXMLImportError(
                "Could not build analysis metadata from the provided CSV input."
            ) from e
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(site_owner = site_owner,
                                           site_description = site_description_dict[siteID],
                                           created = None,
                                           resource_id = siteID)
        if exists_analysis and siteID in analysis_dict:
            sera_site_dict[siteID].analysis = analysis_dict[siteID]

    return sera_site_dict

def excel_to_sera_site(path_or_file_object, velocity_profiles=None):
    """
    Function to import SiteXML metadata from Excel file.
    The Excel file should contain the following sheets:
    - siteOwner: With site ownership metadata. Mandatory.
    - siteDescription: With site description metadata. Mandatory.
    - analysis: With analysis metadata. Optional

    Returns a dictionary of SERASite objects. Dictionary keys are the unique SiteIDs.

    :type path_or_file_object: File name or file like object
    :param path_or_file_object: Excel file with site metadata.
    :type velocity_profiles: str, optional
    :param velocity_profiles: Excel file or path to a folder with velocity profile metadata.
    :rtype: dictionary of :class:`~obspy.io.sitexml.core.SERASite`
    
    Example

    >>> from obspy.io.sitexml.read_csv import excel_to_sera_site
    >>> sera_site_dict = excel_to_sera_site("InputExcel.xlsx",
    ...                         velocity_profiles="vp.xlsx")
    """
    try:
        xls = pd.ExcelFile(path_or_file_object)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access Site metadata Excel file: {path_or_file_object}"
        ) from e
    
    try:
        conv_dict = {
            'velocityS30_year': _read_year_cell,
            'velocityProfile_year': _read_year_cell,
            'resonanceFrequency_year': _read_year_cell,
            'siteClassEC8_year': _read_year_cell,
            'bedrockDepth_year': _read_year_cell,
            'h800_year': _read_year_cell,
            'geologicalUnit_year': _read_year_cell,
        }
        df_dict = pd.read_excel(xls, None, converters=conv_dict)
    except Exception as e:
        raise SiteXMLImportError(
            "Could not read the Excel file with site metadata."
        ) from e

    if not all(k in df_dict for k in ("siteOwner", "siteDescription")):
        raise SiteXMLImportError(
            "The site owner and site description sheets are mandatory."
        )
    
    # Read the velocity profiles and store them
    # in a dictionary of dataframes with key the siteID
    #
    df_vp_dict = _excel_import_velocity_profiles(velocity_profiles) \
        if velocity_profiles else None
    
    #site_owner_dict = _read_sheet(df_dict['siteOwner'], SERASiteOwner)
    site_owner = _read_site_owner(df_dict['siteOwner'])
    site_description_dict = _read_site_description(df_dict['siteDescription'])

    # Read the analysis metadata if the 'analysis' sheet exists.
    #
    if 'analysis' in df_dict:
        exists_analysis = True
        try:
            analysis_dict = _read_analysis(
                df_dict['analysis'], df_vp_dict, skip_invalid_rows=False)
        except Exception as e:
            raise SiteXMLImportError(
                "Could not build analysis metadata from the provided Excel input."
            ) from e
    else:
        warnings.warn("Analysis metadata not provided.", UserWarning)
        exists_analysis = False
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(site_owner = site_owner,
                                           site_description = site_description_dict[siteID],
                                           created = None,
                                           resource_id = siteID)
        if exists_analysis and siteID in analysis_dict:
            sera_site_dict[siteID].analysis = analysis_dict[siteID]

    return sera_site_dict

def _read_site_description(df_site_description):
    """
    Return site-description objects keyed by site ID from tabular metadata.

    :rtype: dict
    """
    
    site_description_dict = {}

    for row in df_site_description.iterrows():

        siteID = _read_cell(row[1], "siteID")
        resource_id =  _read_cell(row[1], "siteDescriptionID")
        latitude = _read_cell(row[1], "latitude")
        longitude = _read_cell(row[1], "longitude")
        if (siteID is None or resource_id is None or
                latitude is None or longitude is None):
            warnings.warn("Missing siteID, siteDescriptionID, latitude or " \
                        "longitude value. " \
                        "Processing of site description element " \
                        "will be skipped.", UserWarning)
            continue
        
        station_code = _read_cell(row[1], "station")
        
        # TODOS
        # If station is empty print a warning
        #
        site_description_obj = SiteDescription(resource_id=resource_id,
                                       station_code=station_code, 
                                       latitude=latitude, 
                                       longitude=longitude)
        
        site_description_obj.altitude = _read_cell(row[1], 
                                            "altitude")
        site_description_obj.min_distance_from_station = _read_cell(row[1], 
                                            "minDistanceFromStation")
        site_description_obj.max_distance_from_station = _read_cell(row[1], 
                                            "maxDistanceFromStation")
        site_description_obj.morphology = _read_cell(row[1], 
                                            "siteMorphology")
        site_description_obj.topographyA = _read_cell(row[1], 
                                            "siteTopography_schemaA")
        site_description_obj.topographyB = _read_cell(row[1], 
                                            "siteTopography_schemaB")
        site_description_obj.preferred_site_analysisID = _read_cell(row[1], 
                                            "preferredSiteAnalysisID")
        site_description_obj.preferred_velocity_profileID = _read_cell(row[1], 
                                            "preferredVelocityProfileID")
        
        site_description_obj.ec8 = \
            _read_site_indicator(row[1], EC8, 'siteClassEC8')
        site_description_obj.bedrock_depth = \
            _read_site_indicator(row[1], BedrockDepth, 'bedrockDepth')
        site_description_obj.h800 = \
            _read_site_indicator(row[1], H800, 'h800')
        site_description_obj.geological_unit = \
            _read_site_indicator(row[1], GeologicalUnit, 'geologicalUnit')
        
        site_description_dict[siteID] = site_description_obj

    return site_description_dict

def _read_analysis(df_analysis, df_vp_dict=None, skip_invalid_rows=True):
    """
    Return a dictionary of Analysis objects for all sites.
    Dictionary key is the siteID.

    :type df_analysis: pandas dataframe, required
    :param df_analysis: Dataframe with analysis metadata for all sites
    :type df_vp_dict: dictionary of pandas dataframes, optional
    :param df_vp_dict: Dictionary of pandas dataframes with velocity
            profile metadata for all sites. Dictionary key is the siteID.
    :rtype: dictionary of :class:`~obspy.io.sitexml.core.Analysis`
    """
    analysis_dict = defaultdict(list)

    for row in df_analysis.iterrows():

        # TODOs What if they don't provide the IDs in the csv file??
        #
        siteID = _read_cell(row[1], "siteID")
        analysisID = _read_cell(row[1], "analysisID")
        site_descriptionID = _read_cell(row[1], "siteDescriptionID")
        #station = row[1]['station']
        if siteID and analysisID and site_descriptionID:
            analysis_obj = Analysis(resource_id = analysisID,
                         site_descriptionID = site_descriptionID)
                
            # Go on reading the site characterization indicators
            analysis_obj.resonance_frequency = \
                _read_site_indicator(row[1], ResonanceFrequency, 'resonanceFrequency')
            analysis_obj.velocity_s30 = \
                _read_site_indicator(row[1], VelocityS30, 'velocityS30')
            analysis_obj.velocity_profile_survey = \
                _read_site_indicator(row[1], VelocityProfileSurvey, 'velocityProfile')
            
            analysis_obj.spt_logs_count = \
                _read_cell(row[1], "sptLogsCount")
            analysis_obj.cpt_logs_count = \
                _read_cell(row[1], "cptLogsCount")
            analysis_obj.borehole_logs_count = \
                _read_cell(row[1], "boreholeLogsCount")
           
            # Read Velocity Profiles of Analysis
            #
            if df_vp_dict and siteID in df_vp_dict and analysis_obj.velocity_profile_survey:
                analysis_obj.velocity_profile_survey.velocity_profiles = \
                    _read_velocity_profiles_for_analysis(
                        df_vp_dict[siteID],
                        analysis_id=analysisID)
            
            # Add analysis object in analysis_dict using as key the siteID
            analysis_dict[siteID].append(analysis_obj)
                                       
        else:
            if skip_invalid_rows:
                warnings.warn("Missing siteID, analysisID or siteDescriptionID "
                              "value. Processing of analysis element will be "
                              "skipped.", UserWarning)
                continue
            raise SiteXMLImportError(
                "Analysis metadata is missing required siteID, analysisID or "
                "siteDescriptionID values. Abording further processing."
            )
    
    return analysis_dict

def _read_velocity_profiles_for_analysis(df_vp, analysis_id):
    """
    Return a list of VelocityProfile objects for a given analysisID.

    :type df_vp: pandas dataframe, required
    :param df_vp: Dataframe of velocity profiles for a single site
    :type analysis_id: str, required
    :param analysis_id: The analysis for which to read velocity profiles
    :rtype: list of :class:`~obspy.io.sitexml.core.velocityProfile`
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
    Build a VelocityProfile object from a subset of rows belonging to a single profile.

    :param rows: A group of dataframe rows
    :rtype: :class:`~obspy.io.sitexml.core.velocityProfile`
    """
    rows = rows.sort_values("layerCount")
    layer_objects = []

    for idx, row in rows.iterrows():

        density = _read_value_with_uncertainty(row, "density")        
        velP = _read_value_with_uncertainty(row, "velocityP")
        velS = _read_value_with_uncertainty(row, "velocityS")
        
        top_depth = _read_value_with_uncertainty(row, "layerTopDepth")
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
    
    if indicator != "velocityProfile":
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
            method_combined_qindex = _read_cell(df_row, 'velocityS30MethodCombIndex')
            manual_qindex = _read_cell(df_row, 'velocityS30ManualIndex')
            obj.method_combined_qindex = (
                None if method_combined_qindex is None else str(method_combined_qindex)
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

    [obj.literature_source, obj.external_reference] = _read_reference(df_row, indicator)
    obj.quality_index = _read_cell(df_row, indicator+'Qindex1')
    
    return obj

def _csv_import_velocity_profiles(path, delim='\t'):
    """
    Read velocity-profile metadata from CSV files or a CSV directory.

    :rtype: dict or None
    """
    return _import_velocity_profiles(
        path=path,
        allowed_extensions=(".csv",),
        read_file=lambda file_path: _read_velocity_profile_csv_file(
            file_path, delim=delim),
        kind_name="CSV")

def _excel_import_velocity_profiles(path):
    """
    Read velocity-profile metadata from Excel files or an Excel directory.

    :rtype: dict or None
    """
    return _import_velocity_profiles(
        path=path,
        allowed_extensions=(".xls", ".xlsx", ".xlsm", ".xlsb"),
        read_file=_read_velocity_profile_excel_file,
        kind_name="Excel")


def _import_velocity_profiles(path, allowed_extensions, read_file, kind_name):
    """
    Read velocity-profile files and return dataframes grouped by site ID.

    :rtype: dict or None
    """
    if not path:
        return None

    path_str = os.fspath(path)
    df = pd.DataFrame()

    if os.path.isdir(path_str):
        for filename in os.listdir(path_str):
            file_path = os.path.join(path_str, filename)
            if not filename.lower().endswith(allowed_extensions):
                raise SiteXMLImportError(
                    f"Velocity-profile input is not a {kind_name} file: {file_path}"
                )
            df = pd.concat([df, read_file(file_path)], ignore_index=True)
    elif os.path.isfile(path_str):
        if not path_str.lower().endswith(allowed_extensions):
            raise SiteXMLImportError(
                f"Velocity-profile input is not a {kind_name} file: {path_str}"
            )
        df = read_file(path_str)
    else:
        raise SiteXMLIOError(f"Velocity-profile path does not exist: {path_str}")

    if not df.empty:
        return {site_id: group for site_id, group in df.groupby("siteID")}
    return None


def _read_velocity_profile_csv_file(file_path, delim='\t'):
    """
    Read one velocity-profile CSV file as a dataframe.

    :rtype: :class:`pandas.DataFrame`
    """
    try:
        return pd.read_csv(file_path, sep=delim)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access velocity-profile CSV file: {file_path}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read velocity-profile CSV file: {file_path}"
        ) from e


def _read_velocity_profile_excel_file(file_path):
    """
    Read all non-empty velocity-profile sheets from one Excel file.

    :rtype: :class:`pandas.DataFrame`
    """
    try:
        df_dict = pd.read_excel(file_path, None)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access velocity-profile Excel file: {file_path}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read velocity-profile Excel file: {file_path}"
        ) from e

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

def _read_reference(df_row, indicator):
    """
    Return literature and external-reference metadata for one indicator.

    :rtype: tuple
    """

    title = _read_cell(df_row, 'title', indicator)
    first_author = _read_cell(df_row, 'firstAuthor', indicator)
    # Title and firstAuthor are required according to the schema.
    #
    if title and first_author:
        literature_source = LiteratureSource(title=title,
                                             first_author=first_author)
        literature_source.secondary_authors = _read_cell(df_row, 'secondaryAuthors', indicator)
        literature_source.year = _read_cell(df_row, 'year', indicator)
        literature_source.booktitle = _read_cell(df_row, 'booktitle', indicator)
        literature_source.language = _read_cell(df_row, 'language', indicator)
        literature_source.doi = _read_cell(df_row, 'doi', indicator)
    elif title or first_author:
        raise SiteXMLImportError(
            f"{indicator} literature source requires both title and "
            "firstAuthor."
        )
    else:
        literature_source = None
    
    uri = _read_cell(df_row, 'uri', indicator)
    description = _read_cell(df_row, 'description', indicator)
    if uri:
        external_reference = ExternalReference(uri = uri,
                                          description = description)
    else:
        external_reference = None

    return literature_source, external_reference

def _read_value_with_uncertainty(row, name):
    """
    Return a ValueWithUncertainty read from ``<name>_value`` columns.

    :rtype: :class:`~obspy.io.sitexml.core.ValueWithUncertainty` or None
    """

    metric = ValueWithUncertainty(row[name+"_value"]) \
        if not _empty_value(row[name+"_value"]) else None
    
    if metric and not _empty_value(row[name+"_uncertainty"]):
        metric.uncertainty = row[name+"_uncertainty"]

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
