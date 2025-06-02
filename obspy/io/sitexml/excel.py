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
import warnings

import pandas as pd

import obspy
from obspy.core.inventory.util import ExternalReference
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, SERASiteOwner, 
                                   EC8, H800, BedrockDepth, GeologicalUnit, ResonanceFrequency, VelocityS30, 
                                   VelocityProfile, VelocityProfileData, ValueWithUncertainty,
                                   LiteratureSource)
from obspy.io.sitexml.write import _write_sitexml


def excel_to_sitexml(sera_site_dict, output_folder):

    for sera_site in sera_site_dict.values():
        print("Creating SiteXMl for station: ", sera_site.station_code)
        
        output_file = output_folder + "/" + sera_site.station_code + ".xml"
        _write_sitexml(sera_site, output_file)

def excel_to_sera_site(path_or_file_object):
    """
    Function import metadata for SiteXML from xlsx files.

    :param path_or_file_object: File name or file like object.

    Returns a dictionary of SERASite objects. Dictionary keys are the station names.
    """

    try:
        xls = pd.ExcelFile(path_or_file_object)
    except FileNotFoundError:
        print(f"Error: File not found: {path_or_file_object}")
        return {}
    
    df_dict = pd.read_excel(xls, None, dtype={'year': str})
    #print(df_dict)
    site_owner_dict = _read_sheet(df_dict['site_owner'], SERASiteOwner)
    site_description_dict = _read_site_description(df_dict)
    site_char_dict = _read_site_characterization(df_dict, site_description_dict.keys())

    if not site_owner_dict or not site_description_dict:
        print("Error: Missing site owner or site description metadata. Aborting...")

    sera_site_dict = {}
    #print(site_description_dict.keys())
    for station in site_description_dict.keys():
        sera_site_dict[station] = SERASite(station_code = station,
                                           site_owner = site_owner_dict['site_owner'],
                                           site_description = site_description_dict[station],
                                           site_characterization = site_char_dict[station],
                                           created = obspy.UTCDateTime())

    
    return sera_site_dict

def _read_site_description(df_dict):
    
    if "site_description" not in df_dict.keys():
        print(f"Error: Missing 'site_description' sheet.")
        return {}
    
    ec8_dict = _read_site_indicator(df_dict, EC8, 'ec8')
    bedrock_depth_dict = _read_site_indicator(df_dict, BedrockDepth, 'bedrock_depth')
    h800_dict = _read_site_indicator(df_dict, H800, 'h800')
    geological_unit_dict = _read_site_indicator(df_dict, GeologicalUnit, 'geological_unit')

    site_description_dict = _read_sheet(df_dict['site_description'], SiteDescription)
    
    for station in site_description_dict.keys():
        site_description_dict[station].ec8 = ec8_dict[station] \
            if station in ec8_dict.keys() else None
        site_description_dict[station].bedrock_depth = \
            bedrock_depth_dict[station] if station in bedrock_depth_dict.keys() else None
        site_description_dict[station].h800 = \
            h800_dict[station] if station in h800_dict.keys() else None
        site_description_dict[station].geological_unit = \
            geological_unit_dict[station] if station in geological_unit_dict.keys() else None

    return site_description_dict

def _read_site_characterization(df_dict, station_list):
    
    #print(df_dict.keys())
    if "site_characterization" not in df_dict.keys():
        print(f"Warning: Missing 'site_characterization' sheet.")
        # Maybe here we should proceed with reading the site indicators
        return {}
    
    rf_dict = _read_site_indicator(df_dict, ResonanceFrequency, 'resonance_frequency')
    vs30_dict = _read_site_indicator(df_dict, VelocityS30, 'velocity_s30')
    site_char_dict = _read_sheet(df_dict['site_characterization'],
                                 SiteCharacterizationParameters)

    # Below it will create an empty SiteCharacterizationParameters object 
    # if site description metadata exists for a station. Maybe it is not desirable
    for station in station_list:
        if station not in site_char_dict.keys():
            site_char_dict[station] = SiteCharacterizationParameters()

        site_char_dict[station].resonance_frequency = \
            rf_dict[station] if station in rf_dict.keys() else None
        site_char_dict[station].velocity_s30 = \
            vs30_dict[station] if station in vs30_dict.keys() else None
    
    return site_char_dict

def _read_site_indicator(df_dict, cls, sheet_name):

    # Check if sheet is missing for a specific site indicator
    if sheet_name not in df_dict.keys():
        return {}
    
    df = df_dict[sheet_name]
    
    # Check if columns 'station_code' and 'value' both exist
    if not {'station_code', 'value'}.issubset(df):
        return {}
    
    # Create a dictionary of the site indicator objects. 
    # Key is the station name
    dict = {}
    for _, row in df.iterrows():

        if not _empty_value(row["value"]) and \
            not _empty_value(row["station_code"]):
            
            obj = cls(value=row["value"])
            if "uncertainty" in df.columns and not _empty_value(row["uncertainty"]):
                obj.value.uncertainty = row["uncertainty"]
            
            for attr in vars(obj).keys():
                if attr != "value" and attr in df.columns and not _empty_value(row[attr]):
                    try:
                        setattr(obj, attr, row[attr])
                    except Exception as e:
                        print(f"Warning: Could not set attribute '{attr}' on {cls.__name__}: {e}")

            # Title is a required attribute for LiteratureSource
            if "title" in df.columns and not _empty_value(row["title"]):
                obj.literature_source = _read_row(df, row, LiteratureSource)
            
            file_resource = _read_row(df, row, ExternalReference)
            if file_resource.uri != "" or file_resource.description != "":
                obj.file_resource = file_resource
            
            dict[row["station_code"]] = obj

    return dict

def _read_sheet(df, cls):

    dict = {}

    for _, row in df.iterrows():
        obj = safe_create_instance(cls)
        
        for attr in vars(obj).keys():
            pub_attr = attr.strip('_')
            if pub_attr in df.columns and not _empty_value(row[pub_attr]):
                try:
                    setattr(obj, pub_attr, row[pub_attr])
                except Exception as e:
                    print(f"Warning: Could not set attribute '{pub_attr}' on {cls.__name__}: {e}")

        if "station_code" in df.columns:
            dict[row["station_code"]] = obj
        else:
            dict["site_owner"] = obj

    return dict

def _read_row(df, row, cls):

    obj = safe_create_instance(cls)

    #print(cls.__name__, vars(obj).keys())
    for attr in vars(obj).keys():
        if attr in df.columns and not _empty_value(row[attr]):
            try:
                #print(attr, type(row[attr]))
                setattr(obj, attr, row[attr])
            except Exception as e:
                print(f"Warning: Could not set attribute '{attr}' on {cls.__name__}: {e}")

    return obj

def safe_create_instance(cls):
    try:
        return cls()  # Try regular constructor
    except TypeError:
        if cls.__name__ == "ExternalReference":
            return cls(uri="", description="")

def _empty_value(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
    
    return False
