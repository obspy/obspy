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
import sys
import re

import pandas as pd
from collections import defaultdict

import obspy
from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import (SERASite, SiteDescription, SERASiteOwner, Analysis,
                                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency, 
                                   VelocityS30, VelocityProfile, VelocityProfileData, 
                                   VelocityProfileSurvey, LiteratureSource, ValueWithUncertainty)
from obspy.io.sitexml.write import _write_sitexml

def csv_to_sitexml(sera_site_dict, output_folder="."):
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
    """
    for sera_site in sera_site_dict.values():
        if sera_site.site_description.station_code:
            print("Creating SiteXMl for station: ", sera_site.site_description.station_code)
            output_file = output_folder + "/" + sera_site.site_description.station_code + ".xml"
        else:
            print("Creating SiteXMl for site: ", sera_site.resource_id.id)
            filename = re.sub(r"[^A-Za-z0-9]+", "_", sera_site.resource_id.id).strip("_")
            output_file = output_folder + "/" + filename + ".xml"
        _write_sitexml(sera_site, output_file, validate=True)

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

    .. rubric:: Example

        >>> from obspy.io.sitexml.csv import csv_to_sera_site
        >>> sera_site_dict = csv_to_sera_site("site_owner.csv", 
                            "site_description.csv", "analysis.csv",
                            "velocity_profiles_dir", ';')
    """
    # This is probably not needed as these two arguments are mandatory.
    #
    if not site_owner_csv or not site_description_csv:
        print("Error: The site owner and site description metadata are madatory. Aborting")
        return None
    
    try:
        df_site_owner = pd.read_csv(site_owner_csv, sep=delim)
        df_site_description = pd.read_csv(site_description_csv, sep=delim)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Try to read the analysis metadata if a csv is provided.
    #
    try:
        df_analysis = pd.read_csv(analysis_csv, sep=delim)
        exists_analysis = True
    except Exception as e:
        print(f"Warning: Missing analysis metadata. {e}")
        exists_analysis = False

    # Read the velocity profiles and store them
    # in a dictionary of dataframes with key the siteID 
    #
    df_vp_dict = _csv_import_velocity_profiles(velocity_profiles_csv, delim=delim)
    
    #site_owner_dict = _read_sheet(df_site_owner, SERASiteOwner)
    site_owner = _read_site_owner(df_site_owner)
    site_description_dict = _read_site_description(df_site_description)

    if exists_analysis:
        analysis_dict = _read_analysis(df_analysis, df_vp_dict)
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(site_owner = site_owner,
                                           site_description = site_description_dict[siteID],
                                           created = obspy.UTCDateTime(),
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
    
    .. rubric:: Example

        >>> from obspy.io.sitexml.csv import excel_to_sera_site
        >>> sera_site_dict = excel_to_sera_site("InputExcel.xlsx", 
                                velocity_profiles="vp.xlsx")
    """
    try:
        xls = pd.ExcelFile(path_or_file_object)
    except FileNotFoundError:
        print(f"Error: File not found: {path_or_file_object}")
        return {}
    
    try:
        conv_dict = {'velocityS30_year': int, 'velocityProfile_year': int,
                     'resonanceFrequency_year': int, 'siteClassEC8_year': int,
                     'bedrockDepth_year': int, 'h800_year': int,
                     'geologicalUnit_year': int}
        df_dict = pd.read_excel(xls, None, converters=conv_dict)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not all(k in df_dict for k in ("siteOwner", "siteDescription")):
        print("Error: The site owner and site description metadata are madatory. Aborting")
        return None
    
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
        analysis_dict = _read_analysis(df_dict['analysis'], df_vp_dict)
    else:
        print("Warning: Missing analysis metadata.")
        exists_analysis = False
        
    # All dictionaries use the unique SiteID for key.
    #
    sera_site_dict = {}
    for siteID in site_description_dict:
        sera_site_dict[siteID] = SERASite(site_owner = site_owner,
                                           site_description = site_description_dict[siteID],
                                           created = obspy.UTCDateTime(),
                                           resource_id = siteID)
        if exists_analysis and siteID in analysis_dict:
            sera_site_dict[siteID].analysis = analysis_dict[siteID]

    return sera_site_dict

def _read_site_description(df_site_description):
    
    site_description_dict = {}

    for row in df_site_description.iterrows():

        siteID = _read_cell(row[1], "siteID")
        latitude = _read_cell(row[1], "latitude")
        longitude = _read_cell(row[1], "longitude")
        if siteID is None or latitude is None or longitude is None:
            warnings.warn("Missing siteID, latitude or longitude value. " \
                        "Processing of site description element " \
                        "will be skipped.", UserWarning)
            return None
        
        # TODOs What if they don't provide the IDs in the csv file??
        #
        resource_id =  _read_cell(row[1], "siteDescriptionID")
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

def _read_analysis(df_analysis, df_vp_dict=None):
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
        siteID = row[1]['siteID']
        analysisID = row[1]['analysisID']
        site_descriptionID = row[1]['siteDescriptionID']
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
            
            analysis_obj.velocity_profile_count = \
                _read_cell(row[1], "velocityProfileCount")
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
            return None
    
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
            density=density,
            velocityP=velP,
            velocityS=velS,
            top_depth=top_depth,
            bottom_depth=bottom_depth
        )

        layer_objects.append(layer_obj)

    return VelocityProfile(
        layer_count=len(layer_objects),
        resource_id=rows.iloc[0]["velocityProfileID"],
        velocity_profile_data=layer_objects
    )

def _read_site_indicator(df_row, cls, indicator):
    
    if indicator != "velocityProfile":
        if _empty_value(df_row[indicator+'_value']):
            return None
        
        obj = cls(value=df_row[indicator+'_value'])
        
        if _read_cell(df_row, indicator+'_uncertainty'):
            obj.value.uncertainty = df_row[indicator+'_uncertainty']

        if _read_cell(df_row, indicator+'Method1'):
            obj.methods.append(df_row[indicator+'Method1'])
        if _read_cell(df_row, indicator+'Method2'):
            obj.methods.append(df_row[indicator+'Method2'])
        
        if indicator == "velocityS30":
            obj.method_combined_quality_index = _read_cell(df_row, 'velocityS30MethodCombIndex')
            obj.manual_quality_index = _read_cell(df_row, 'velocityS30ManualIndex')
        
        if indicator == "geologicalUnit":
            obj.geological_map_scale = _read_cell(df_row, 'geologicalMapScale')
            obj.geological_unit_OGE = _read_cell(df_row, 'geologicalUnitOGE')
    else:
        obj = cls()

    [obj.literature_source, obj.file_resource] = _read_reference(df_row, indicator)
    obj.quality_index = _read_cell(df_row, indicator+'Qindex1')
    
    return obj

def _csv_import_velocity_profiles(path, delim='\t'):
    
    df = pd.DataFrame()

    # Case 1: path is a directory → loop through CSV files
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(path, filename)
                try:
                    df = pd.concat([df, pd.read_csv(file_path, sep=delim)], ignore_index=True)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # Case 2: path is a file → read only that file
    elif os.path.isfile(path):
        if path.lower().endswith(".csv"):
            try:
                df = pd.read_csv(path, sep=delim)
            except Exception as e:
                print(f"Error reading file {path}: {e}")
        else:
            print(f"File is not a CSV: {path}")

    # Case 3: path is invalid
    else:
        print(f"Invalid path: {path}")

    if not df.empty:
        df_dict = {site_id: group for site_id, group in df.groupby("siteID")}
        return df_dict 
    else:
        return None

def _excel_import_velocity_profiles(path):
    
    df = pd.DataFrame()

    def _read_excel_file(file_path):
        nonlocal df
        try:
            df_dict = pd.read_excel(file_path, None)
        except Exception as e:
            print(f"Error reading excel file {file_path}: {e}")
            return
        
        for sheet_name, sheet_df in df_dict.items():
            if sheet_df is None or sheet_df.empty:
                continue
            if "siteID" not in sheet_df.columns:
                print(f"Error: Missing 'siteID' column in sheet {sheet_name} of {file_path}")
                raise ValueError("Missing required 'siteID' column.")
            df = pd.concat([df, sheet_df], ignore_index=True)

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                try:
                    _read_excel_file(os.path.join(path, filename))
                except ValueError:
                    return None

    elif os.path.isfile(path):
        if path.lower().endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
            try:
                _read_excel_file(path)
            except ValueError:
                return None
        else:
            print(f"File is not an Excel file: {path}")

    else:
        print(f"Invalid path: {path}")

    if not df.empty:
        df_dict = {site_id: group for site_id, group in df.groupby("siteID")}
        return df_dict

    return None

def _read_reference(df_row, indicator):

    title = _read_cell(df_row, 'title', indicator)
    # Title is the only required property according to schema
    #
    if title: 
        literature_source = LiteratureSource(title = title)
        literature_source.first_author = _read_cell(df_row, 'firstAuthor', indicator)
        literature_source.secondary_authors = _read_cell(df_row, 'secondaryAuthors', indicator)
        literature_source.year = _read_cell(df_row, 'year', indicator)
        literature_source.booktitle = _read_cell(df_row, 'booktitle', indicator)
        literature_source.language = _read_cell(df_row, 'language', indicator)
        literature_source.doi = _read_cell(df_row, 'doi', indicator)
    else:
        literature_source = None
    
    uri = _read_cell(df_row, 'url', indicator)
    description = _read_cell(df_row, 'description', indicator)
    if uri:
        file_resource = ExternalReference(uri = uri,
                                          description = description)
    else:
        file_resource = None

    return literature_source, file_resource

def _read_value_with_uncertainty(row, name):

    metric = ValueWithUncertainty(row[name+"_value"]) \
        if not _empty_value(row[name+"_value"]) else None
    
    if metric and not _empty_value(row[name+"_uncertainty"]):
        metric.uncertainty = row[name+"_uncertainty"]

    return metric

def _read_site_owner(df):

    obj = SERASiteOwner()

    for _, row in df.iterrows():
        #obj = _safe_create_instance(cls)
        
        for attr in vars(obj).keys():
            pub_attr = attr.strip('_')
            if pub_attr in df.columns and not _empty_value(row[pub_attr]):
                try:
                    setattr(obj, pub_attr, row[pub_attr])
                except Exception as e:
                    print(f"Warning: Could not set attribute '{pub_attr}' on SiteOwner: {e}")
    return obj

def _read_cell(df_row, argument, indicator=None):

    if indicator:
        if indicator + "_" + argument in df_row and \
            not _empty_value(df_row[indicator + "_" + argument]):
                    return df_row[indicator + "_" + argument]
    else:
        if argument in df_row and \
            not _empty_value(df_row[argument]):
                    return df_row[argument]
        
    return None

def _empty_value(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
    
    return False