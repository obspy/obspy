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

import pandas as pd
from collections import defaultdict

import obspy
from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, SERASiteOwner, 
                                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency, VelocityS30, 
                                   VelocityProfile, VelocityProfileData, ValueWithUncertainty, Analysis,
                                   LiteratureSource, VelocityProfileSurvey)
from obspy.io.sitexml.write import _write_sitexml
from obspy.io.sitexml.excel import _read_sheet, _empty_value


def csv_to_sitexml(sera_site_dict, output_folder):

    for sera_site in sera_site_dict.values():
        print("Creating SiteXMl for station: ", sera_site.site_description.station_code)
        
        output_file = output_folder + "/" + sera_site.site_description.station_code + ".xml"
        _write_sitexml(sera_site, output_file)

def csv_to_sera_site(site_owner_csv,
                     site_description_csv, 
                     site_char_csv=None, 
                     velocity_profile_dir=None, 
                     delim='\t'):
    """
    Function import metadata for SiteXML from csv files.

    :type site_owner_csv: File name or file like object
    :param site_owner_csv: One line csv file with site owner metadata. Mandatory.
    :type site_description_csv: File name or file like object
    :param site_description_csv: CSV file with site description metadata. 
            One line per station/location. Mandatory.
    :type site_char_csv: File name or file like object
    :param site_char_csv: CSV file with site characterization metadata. 
            One line per analysisID. Optional.
    :type velocity_profile_dir: 
    :param velocity_profile_dir: Path to a folder with velocity profile metadata.
            The folder should contain one CSV file per station. Optional.
    :type delim: str
    :param delim: CSV file delimiter. Default tab delimeted.

    Returns a dictionary of SERASite objects. Dictionary keys are the station names.
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
    
    # Try to read the site characterization metadata if a csv is provided.
    #
    try:
        df_site_char = pd.read_csv(site_char_csv, sep=delim)
        exists_site_char = True
    except Exception as e:
        print(f"Warning: Missing Site Characterization metadata. {e}")
        exists_site_char = False

    # Read the velocity profiles and 
    # Store them in a dictionary of dataframes with key the station name 
    #
    df_vp_dict, errors = _load_csv_directory(velocity_profile_dir, delim=delim)
    #print("Velocity profiles keys: ", df_vp_dict.keys())
    
    #df_site_owner.info()
    #df_site_description.info()
    
    site_owner_dict = _read_sheet(df_site_owner, SERASiteOwner)
    site_description_dict = _read_site_description(df_site_description)
    #print("site_description_dict keys: ")
    #for key in site_description_dict:
    #    print(key)

    if exists_site_char:
        #df_site_char.info()
        site_char_dict = _read_site_characterization(df_site_char, df_vp_dict)
        #print("site_char_dict keys: ", site_char_dict.keys())
        
    # All dictionaries use the station name for key.
    # What if the sitexml is for a site other than a station?
    #
    sera_site_dict = {}
    for key in site_description_dict:
        sera_site_dict[key] = SERASite(site_owner = site_owner_dict['site_owner'],
                                           site_description = site_description_dict[key],
                                           created = obspy.UTCDateTime())
        if exists_site_char and key in site_char_dict:
            sera_site_dict[key].site_characterization = site_char_dict[key]
        """
        site_descriptionID = site_description_dict[key].publicID
        print(site_descriptionID)
        if exists_site_char and site_descriptionID in site_char_dict:
            sera_site_dict[key].site_characterization = site_char_dict[site_descriptionID]
        """

    return sera_site_dict

def _read_site_description(df_site_description):
    
    site_description_dict = {}

    #for row in df_site_description.i
    for i in range(df_site_description.ndim):
        #print(df_site_description[i]['station'])

        latitude = df_site_description.loc[i].at['latitude']
        longitude = df_site_description.loc[i].at['longitude']
        if latitude is None or longitude is None:
            warnings.warn("Missing latitude or longitude value. " \
                        "Processing of site description element " \
                        "will be skipped.", UserWarning)
            return None
        
        # TODOs What if they don't provide the IDs in the csv file??
        #
        publicID =  _read_cell(df_site_description.loc[i],"siteDescriptionID")
        station_code = _read_cell(df_site_description.loc[i],"station")
        
        # TODOS
        # If station is empty print a warning
        #
        site_description_obj = SiteDescription(publicID=publicID,
                                       station_code=station_code, 
                                       latitude=latitude, 
                                       longitude=longitude)
        
        site_description_obj.altitude = \
            _read_cell(df_site_description.loc[i], 
                                            "altitude")
        site_description_obj.min_distance_from_station = _read_cell(df_site_description.loc[i], 
                                            "minDistanceFromStation")
        site_description_obj.max_distance_from_station = _read_cell(df_site_description.loc[i], 
                                            "maxDistanceFromStation")
        site_description_obj.morphology = _read_cell(df_site_description.loc[i], 
                                            "siteMorphology")
        site_description_obj.topographyA = _read_cell(df_site_description.loc[i], 
                                            "siteTopography_schemaA")
        site_description_obj.topographyB = _read_cell(df_site_description.loc[i], 
                                            "siteTopography_schemaB")
        site_description_obj.preferred_site_analysisID = _read_cell(df_site_description.loc[i], 
                                            "preferredSiteAnalysisID")
        site_description_obj.preferred_velocity_profileID = _read_cell(df_site_description.loc[i], 
                                            "preferredVelocityProfileID")
        
        site_description_obj.ec8 = \
            _read_site_indicator(df_site_description.loc[i], EC8, 'siteClassEC8')
        site_description_obj.bedrock_depth = \
            _read_site_indicator(df_site_description.loc[i], BedrockDepth, 'bedrockDepth')
        site_description_obj.h800 = \
            _read_site_indicator(df_site_description.loc[i], H800, 'h800')
        site_description_obj.geological_unit = \
            _read_site_indicator(df_site_description.loc[i], GeologicalUnit, 'geologicalUnit')
        
        # TODOs 
        # if station code is missing we need another index for dictionary
        site_description_dict[station_code] = site_description_obj

    return site_description_dict

def _read_site_characterization(df_site_char, df_vp_dict=None):
    
    site_char_dict = {}
    analysis_dict = defaultdict(list)

    for row in df_site_char.iterrows():

        # At least one analysis should exist in order to create the SiteChar object
        #
        # TODOs What if they don't provide the IDs in the csv file??
        #
        siteCharacterizationID = row[1]['siteCharacterizationID']
        analysisID = row[1]['analysisID']
        site_descriptionID = row[1]['siteDescriptionID']
        station = row[1]['station']
        if siteCharacterizationID and analysisID and site_descriptionID:
            analysis_obj = Analysis(publicID = analysisID,
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
            if analysis_obj.velocity_profile_survey and station in df_vp_dict:
                analysis_obj.velocity_profile_survey.velocity_profiles = \
                    _read_velocity_profiles_for_analysis(
                        df_vp_dict[station],
                        analysis_id=analysisID)
            
            # Add analysis object in analysis_dict using as key the station name
            analysis_dict[station].append(analysis_obj)
            
            # Create a new SiteCharacterizationParameters obj and append to the dictionary 
            # only if this the row we encounter first occurence of the siteCharacterization object
            # Use as key the site description ID so we can associate with the site description metadata
            if site_descriptionID not in site_char_dict:
                site_char_dict[station] = \
                    SiteCharacterizationParameters(publicID = siteCharacterizationID)                                
        else:
            return None
    
    # Cycle throught the analyis dict and 
    # assign the analysis objects in the appropriate site_characterization objects
    #
    for st in analysis_dict:
        for analysis in analysis_dict[st]:
            site_char_dict[st].analysis.append(analysis)

    return site_char_dict

def _read_velocity_profiles_for_analysis(df_vp, analysis_id):
    """
    Return a list of VelocityProfile objects for a given analysisID.

    :type df_vp: Pandas dataframe 
    :param df_vp: Dataframe of velocity profiles for a station
    :type analysis_id: str
    :param analysis_id: The analysis for which to read velocity profiles
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
        publicID=rows.iloc[0]["velocityProfileID"],
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
    obj.quality_index = _read_quality_index(df_row, indicator)

    return obj

def _load_csv_directory(path, delim='\t'):
    # Check whether the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Check whether the path is a directory
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")

    dataframes = {}
    errors = {}

    for filename in os.listdir(path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(path, filename)
            key = os.path.splitext(filename)[0]

            try:
                df = pd.read_csv(file_path, sep=delim)
                dataframes[key] = df
            except Exception as e:
                # Store the error so user knows which files failed
                errors[key] = str(e)

    return dataframes, errors    

def _read_reference(df_row, indicator):

    title = _read_cell(df_row, 'title', indicator)
    # TODOs
    # first_author is also required according to the schema
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
    if uri or description:
        file_resource = ExternalReference(uri = uri,
                                          description = description)
    else:
        file_resource = None

    return literature_source, file_resource

def _read_quality_index(df_row, indicator):
    if _read_cell(df_row, indicator+'Qindex1_value'):
        if _read_cell(df_row, indicator+'Qindex1_uncertainty'):
            vwu = ValueWithUncertainty(value = df_row[indicator+'Qindex1_value'],
                                        uncertainty = df_row[indicator+'Qindex1_uncertainty'])
        else:
            vwu = ValueWithUncertainty(value = df_row[indicator+'Qindex1_value'])
        return vwu
    else:       
        return None

def _read_value_with_uncertainty(row, name):

    metric = ValueWithUncertainty(row[name+"_value"]) \
        if not _empty_value(row[name+"_value"]) else None
    
    if metric and not _empty_value(row[name+"_uncertainty"]):
        metric.uncertainty = row[name+"_uncertainty"]

    return metric

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
